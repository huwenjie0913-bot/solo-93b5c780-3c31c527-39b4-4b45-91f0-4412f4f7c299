"""End-to-end tests for the FastAPI coordination endpoints.

Run:
    python -m pip install -r requirements.txt
    pytest -q
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()
    monkeypatch.setenv("RF_DB_PATH", tmp.name)
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    with TestClient(main.app) as c:
        yield c
    os.unlink(tmp.name)


def device(name: str, **overrides):
    data = {
        "name": name,
        "kind": "mic",
        "band_min_mhz": 470.0,
        "band_max_mhz": 472.0,
        "step_mhz": 0.1,
        "bandwidth_mhz": 0.1,
        "priority": 1,
        "tx_power_dbm": 20,
    }
    data.update(overrides)
    return data


def test_create_list_export_and_compare(client):
    payload = {
        "name": "show",
        "devices": [device("mic a", priority=5), device("mic b")],
        "scan_zones": [
            {"name": "occupied", "start_mhz": 471.0, "end_mhz": 471.2, "power_dbm": -30}
        ],
        "max_search_nodes": 1000,
    }
    created = client.post("/api/coordination/plans", json=payload)
    assert created.status_code == 201, created.text
    plan_a = created.json()
    assert plan_a["scheme"]["total_conflict_count"] >= 0
    assert len(plan_a["assignments"]) == 2
    assert plan_a["device_reports"][0]["nearest_interference_source"] is not None

    exported = client.get(f"/api/coordination/plans/{plan_a['plan_id']}/export")
    assert exported.status_code == 200
    assert exported.json()["result"]["assignments"] == plan_a["assignments"]

    revised_payload = {
        "additional_scan_zones": [
            {"name": "new blocker", "start_mhz": 470.4, "end_mhz": 470.6, "power_dbm": -20}
        ]
    }
    revised = client.post(
        f"/api/coordination/plans/{plan_a['plan_id']}/recalculate", json=revised_payload
    )
    assert revised.status_code == 201, revised.text
    plan_b = revised.json()
    assert plan_b["parent_plan_id"] == plan_a["plan_id"]

    comparison = client.post(
        "/api/coordination/compare",
        json={"plan_id_a": plan_a["plan_id"], "plan_id_b": plan_b["plan_id"]},
    )
    assert comparison.status_code == 200
    assert "conflict_count_b_minus_a" in comparison.json()["delta"]

    history = client.get("/api/coordination/plans")
    assert history.status_code == 200
    assert {p["id"] for p in history.json()["plans"]} >= {
        plan_a["plan_id"],
        plan_b["plan_id"],
    }


def test_lock_off_grid_identifies_device(client):
    payload = {
        "name": "bad lock",
        "devices": [device("mic a", locked_mhz=470.05)],
    }
    response = client.post("/api/coordination/plans", json=payload)
    assert response.status_code == 409
    body = response.json()["error"]
    assert body["code"] == "CONSTRAINT_CONFLICT"
    assert body["details"][0]["device_index"] == 0
    assert body["details"][0]["code"] == "LOCK_NOT_TUNABLE"


def test_locked_pair_conflict_identifies_both_devices(client):
    payload = {
        "name": "locked pair",
        "devices": [
            device("mic a", locked_mhz=470.0),
            device("mic b", locked_mhz=470.05),
        ],
    }
    response = client.post("/api/coordination/plans", json=payload)
    assert response.status_code == 409
    detail = response.json()["error"]["details"][0]
    assert detail["code"] == "LOCKED_ADJACENT_CONFLICT"
    assert detail["device_indices"] == [0, 1]
    assert detail["required_separation_hz"] > detail["separation_hz"]


def test_cross_band_second_harmonic_avoids_future_victim_conflict(client):
    rule_response = client.post(
        "/api/rules",
        json={
            "name": "cross-band-im-regression",
            "guard_spacing_mhz": 0.001,
            "adjacent_isolation_db": 0,
            "coupling_loss_db": 0,
            "im2_rejection_db": 0,
            "im3_rejection_db": 0,
            "victim_threshold_dbm": -200,
        },
    )
    assert rule_response.status_code == 201, rule_response.text
    rule_id = rule_response.json()["id"]

    def cross_band_devices(a_locked=None, c_locked=None):
        return [
            device(
                "UHF TX",
                band_min_mhz=530.2,
                band_max_mhz=530.8,
                step_mhz=0.6,
                bandwidth_mhz=0.01,
                tx_power_dbm=20,
                locked_mhz=a_locked,
            ),
            device(
                "VHF TX",
                band_min_mhz=100.2,
                band_max_mhz=100.3,
                step_mhz=0.1,
                bandwidth_mhz=0.01,
                tx_power_dbm=20,
                locked_mhz=100.2,
            ),
            device(
                "future victim",
                band_min_mhz=200.3,
                band_max_mhz=200.4,
                step_mhz=0.1,
                bandwidth_mhz=0.01,
                tx_power_dbm=20,
                locked_mhz=c_locked,
            ),
        ]

    # Small exhaustive reference: only UHF and the future victim have choices.
    exhaustive = {}
    for a_frequency, c_frequency in [
        (530.2, 200.3),
        (530.2, 200.4),
        (530.8, 200.3),
        (530.8, 200.4),
    ]:
        locked_response = client.post(
            "/api/coordination/plans",
            json={
                "name": f"exhaustive {a_frequency} {c_frequency}",
                "rule_id": rule_id,
                "devices": cross_band_devices(a_frequency, c_frequency),
                "max_search_nodes": 100,
            },
        )
        assert locked_response.status_code == 201, locked_response.text
        plan = locked_response.json()
        exhaustive[(a_frequency, c_frequency)] = plan["scheme"]["total_conflict_count"]

    assert exhaustive == {
        (530.2, 200.3): 0,
        (530.2, 200.4): 1,
        (530.8, 200.3): 0,
        (530.8, 200.4): 1,
    }

    bad_locked = client.post(
        "/api/coordination/plans",
        json={
            "name": "known bad harmonic assignment",
            "rule_id": rule_id,
            "devices": cross_band_devices(530.8, 200.4),
            "max_search_nodes": 100,
        },
    )
    assert bad_locked.status_code == 201
    victim_report = bad_locked.json()["device_reports"][2]
    assert [
        threat
        for threat in victim_report["intermodulation_threats"]
        if threat["is_conflict"]
        and threat["kind"] == "IM2:2f"
        and threat["frequency_mhz"] == 200.4
        and threat["source_device_indices"] == [1]
    ]

    optimized = client.post(
        "/api/coordination/plans",
        json={
            "name": "cross-band optimized",
            "rule_id": rule_id,
            "devices": cross_band_devices(),
            "max_search_nodes": 100,
        },
    )
    assert optimized.status_code == 201, optimized.text
    result = optimized.json()
    assert result["scheme"]["total_conflict_count"] == min(exhaustive.values())
    assert result["scheme"]["weighted_conflict_count"] == 0
    assert [a["frequency_mhz"] for a in result["assignments"]] == [530.2, 100.2, 200.3]
    assert all(
        report["conflict_count"] == 0 for report in result["device_reports"]
    )


def test_invalid_step_validation(client):
    payload = {
        "name": "invalid step",
        "devices": [device("mic a", step_mhz=0)],
    }
    response = client.post("/api/coordination/plans", json=payload)
    assert response.status_code == 422
    detail = response.json()["error"]["details"][0]
    assert "step_mhz" in detail["location"]
    assert detail.get("device_index") == 0
