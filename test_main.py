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
