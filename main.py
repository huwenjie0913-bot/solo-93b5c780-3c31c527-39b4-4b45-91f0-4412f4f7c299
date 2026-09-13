"""
Radio-frequency coordination service.

Frequency units are MHz on the HTTP boundary.  Internally every frequency and
spacing is converted to integer Hz to avoid floating-point grid errors such as
470.1 MHz not being exactly representable in binary floating point.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

MHZ_TO_HZ = 1_000_000
MAX_CANDIDATE_POINTS = 200_000
DEFAULT_MAX_SEARCH_NODES = 20_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def mhz_to_hz(value: float) -> int:
    try:
        return int((Decimal(str(value)) * MHZ_TO_HZ).to_integral_value())
    except (InvalidOperation, ValueError):
        return int(round(float(value) * MHZ_TO_HZ))


def hz_to_mhz(value: int) -> float:
    # Integer Hz means at most six useful decimal places.
    return round(value / MHZ_TO_HZ, 6)


# ---------------------------------------------------------------------------
# HTTP models
# ---------------------------------------------------------------------------


class DeviceSpec(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    kind: str = Field(..., pattern="^(mic|iem|intercom|other)$")
    band_min_mhz: float = Field(..., ge=0.0, le=100_000.0)
    band_max_mhz: float = Field(..., ge=0.0, le=100_000.0)
    step_mhz: float = Field(..., gt=0.0, le=10_000.0)
    bandwidth_mhz: float = Field(..., gt=0.0, le=1000.0)
    priority: int = Field(1, ge=1, le=5)
    tx_power_dbm: float = Field(20.0, ge=-60.0, le=60.0)
    locked_mhz: Optional[float] = Field(None, ge=0.0, le=100_000.0)


class ScanZone(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    start_mhz: float = Field(..., ge=0.0, le=100_000.0)
    end_mhz: float = Field(..., ge=0.0, le=100_000.0)
    power_dbm: float = Field(-20.0, ge=-100.0, le=80.0)


class RuleInput(BaseModel):
    name: str = Field("default", min_length=1, max_length=100)
    guard_spacing_mhz: float = Field(0.025, gt=0.0, le=10.0)
    adjacent_isolation_db: float = Field(10.0, ge=0.0, le=120.0)
    coupling_loss_db: float = Field(20.0, ge=0.0, le=120.0)
    im2_rejection_db: float = Field(90.0, ge=0.0, le=200.0)
    im3_rejection_db: float = Field(120.0, ge=0.0, le=200.0)
    victim_threshold_dbm: float = Field(-100.0, ge=-160.0, le=50.0)


class CoordinationRequest(BaseModel):
    name: str = Field("untitled plan", min_length=1, max_length=100)
    devices: List[DeviceSpec] = Field(..., min_length=1, max_length=100)
    scan_zones: List[ScanZone] = Field(default_factory=list)
    rule_id: Optional[int] = None
    note: str = Field("", max_length=1000)
    max_search_nodes: int = Field(DEFAULT_MAX_SEARCH_NODES, ge=1, le=1_000_000)


class RecalculateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    replacements: Dict[int, DeviceSpec] = Field(default_factory=dict)
    additional_scan_zones: List[ScanZone] = Field(default_factory=list)
    # Locks listed here override the old request; unlocked_indices is applied
    # last so it can explicitly clear conflicting override values.
    locked_mhz: Dict[int, Optional[float]] = Field(default_factory=dict)
    unlocked_indices: List[int] = Field(default_factory=list)
    rule_id: Optional[int] = None
    note: str = Field("", max_length=1000)
    max_search_nodes: int = Field(DEFAULT_MAX_SEARCH_NODES, ge=1, le=1_000_000)


class CompareRequest(BaseModel):
    plan_id_a: str
    plan_id_b: str


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class APIError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Optional[List[Dict[str, Any]]] = None,
    ):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or []
        super().__init__(message)


def error_response(status_code: int, code: str, message: str, details=None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "details": details or []}},
    )


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


DB_PATH = os.environ.get("RF_DB_PATH", os.path.join(os.getcwd(), "rf_coordination.db"))


def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS rule_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            rules_json TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS plans (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            rule_id INTEGER,
            request_json TEXT NOT NULL,
            result_json TEXT NOT NULL,
            parent_plan_id TEXT,
            note TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            conflict_count INTEGER NOT NULL,
            weighted_conflict_count REAL NOT NULL,
            min_frequency_margin_hz INTEGER NOT NULL,
            min_level_margin_db REAL NOT NULL,
            FOREIGN KEY(rule_id) REFERENCES rule_versions(id),
            FOREIGN KEY(parent_plan_id) REFERENCES plans(id)
        );

        CREATE INDEX IF NOT EXISTS idx_plans_created_at ON plans(created_at);
        CREATE INDEX IF NOT EXISTS idx_rules_active ON rule_versions(active);
        """
    )
    count = conn.execute("SELECT COUNT(*) FROM rule_versions").fetchone()[0]
    if count == 0:
        rules = RuleInput(name="default-v1").model_dump()
        cur = conn.execute(
            "INSERT INTO rule_versions(name, rules_json, active, created_at) VALUES (?, ?, 1, ?)",
            ("default-v1", json.dumps(rules, ensure_ascii=False), utc_now()),
        )
        conn.commit()
        rules["id"] = cur.lastrowid


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = connect_db()
    init_db(app.state.db)
    yield
    app.state.db.close()


app = FastAPI(
    title="Live RF Coordination API",
    version="1.0.0",
    description="Wireless microphone, IEM and intercom frequency coordination.",
    lifespan=lifespan,
)


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return error_response(exc.status_code, exc.code, exc.message, exc.details)


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(request: Request, exc: RequestValidationError):
    details: List[Dict[str, Any]] = []
    for err in exc.errors():
        loc = [str(x) for x in err.get("loc", []) if x != "body"]
        device_index = None
        if "devices" in loc:
            for part in loc:
                if part.isdigit():
                    device_index = int(part)
                    break
        detail = {
            "location": loc,
            "message": err.get("msg"),
            "constraint": err.get("type"),
        }
        if device_index is not None:
            detail["device_index"] = device_index
        details.append(detail)
    return error_response(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "VALIDATION_FAILED",
        "Request parameters are invalid.",
        details,
    )


@app.exception_handler(ValidationError)
async def pydantic_error_handler(request: Request, exc: ValidationError):
    return validation_error_response(exc)


def validation_error_response(exc: ValidationError) -> JSONResponse:
    details: List[Dict[str, Any]] = []
    for err in exc.errors():
        loc = [str(x) for x in err.get("loc", [])]
        device_index = None
        if "devices" in loc:
            for part in loc:
                if part.isdigit():
                    device_index = int(part)
                    break
        detail = {
            "location": loc,
            "message": err.get("msg"),
            "constraint": err.get("type"),
        }
        if device_index is not None:
            detail["device_index"] = device_index
        details.append(detail)
    return error_response(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "VALIDATION_FAILED",
        "Request parameters are invalid.",
        details,
    )


def get_db(request: Request) -> sqlite3.Connection:
    return request.app.state.db


def fetch_rule(conn: sqlite3.Connection, rule_id: Optional[int]) -> sqlite3.Row:
    if rule_id is not None:
        row = conn.execute("SELECT * FROM rule_versions WHERE id = ?", (rule_id,)).fetchone()
        if row is None:
            raise APIError(404, "RULE_NOT_FOUND", f"Rule version {rule_id} does not exist.")
        return row
    row = conn.execute(
        "SELECT * FROM rule_versions WHERE active = 1 ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise APIError(500, "NO_ACTIVE_RULE", "No active rule version exists.")
    return row


def rule_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    data = json.loads(row["rules_json"])
    data["id"] = row["id"]
    data["active"] = bool(row["active"])
    data["created_at"] = row["created_at"]
    return data


# ---------------------------------------------------------------------------
# Normalization and candidate generation
# ---------------------------------------------------------------------------


def normalize_devices(devices: List[DeviceSpec]) -> List[Dict[str, Any]]:
    normalized = []
    for i, d in enumerate(devices):
        band_min = mhz_to_hz(d.band_min_mhz)
        band_max = mhz_to_hz(d.band_max_mhz)
        if band_min >= band_max:
            raise APIError(
                422,
                "INVALID_BAND",
                f"Device {i} has an empty frequency band.",
                [
                    {
                        "device_index": i,
                        "device_name": d.name,
                        "constraint": "band_min_mhz < band_max_mhz",
                        "band_min_mhz": d.band_min_mhz,
                        "band_max_mhz": d.band_max_mhz,
                    }
                ],
            )
        normalized.append(
            {
                "index": i,
                "name": d.name,
                "kind": d.kind,
                "band_min": band_min,
                "band_max": band_max,
                "step": mhz_to_hz(d.step_mhz),
                "bandwidth": mhz_to_hz(d.bandwidth_mhz),
                "half_bandwidth": (mhz_to_hz(d.bandwidth_mhz) + 1) // 2,
                "priority": d.priority,
                "power_dbm": float(d.tx_power_dbm),
                "locked": mhz_to_hz(d.locked_mhz) if d.locked_mhz is not None else None,
            }
        )
    return normalized


def normalize_scan_zones(zones: List[ScanZone]) -> List[Dict[str, Any]]:
    result = []
    for i, z in enumerate(zones):
        start = mhz_to_hz(z.start_mhz)
        end = mhz_to_hz(z.end_mhz)
        if start >= end:
            raise APIError(
                422,
                "INVALID_SCAN_ZONE",
                f"Scan zone {i} has an empty interval.",
                [
                    {
                        "scan_zone_index": i,
                        "scan_zone_name": z.name,
                        "constraint": "start_mhz < end_mhz",
                    }
                ],
            )
        result.append(
            {
                "index": i,
                "name": z.name,
                "start": start,
                "end": end,
                "power_dbm": float(z.power_dbm),
            }
        )
    return result


def normalize_rules(rule: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": rule["id"],
        "name": rule["name"],
        "guard_spacing": mhz_to_hz(rule["guard_spacing_mhz"]),
        "adjacent_isolation_db": float(rule["adjacent_isolation_db"]),
        "coupling_loss_db": float(rule["coupling_loss_db"]),
        "im2_rejection_db": float(rule["im2_rejection_db"]),
        "im3_rejection_db": float(rule["im3_rejection_db"]),
        "victim_threshold_dbm": float(rule["victim_threshold_dbm"]),
    }


def grid_points(device: Dict[str, Any]) -> List[int]:
    start, stop, step = device["band_min"], device["band_max"], device["step"]
    if step <= 0:
        # Pydantic normally catches this, but keep the internal invariant explicit.
        raise APIError(
            422,
            "INVALID_STEP",
            f"Device {device['index']} has an illegal channel step.",
            [{"device_index": device["index"], "constraint": "step_mhz > 0"}],
        )
    last = start + ((stop - start) // step) * step
    if last < start:
        return []
    count = (last - start) // step + 1
    if count > MAX_CANDIDATE_POINTS:
        raise APIError(
            422,
            "TOO_MANY_CANDIDATES",
            f"Device {device['index']} has more than {MAX_CANDIDATE_POINTS} grid points.",
            [
                {
                    "device_index": device["index"],
                    "candidate_count": int(count),
                    "limit": MAX_CANDIDATE_POINTS,
                    "constraint": "reduce band width or increase channel step",
                }
            ],
        )
    return [start + k * step for k in range(int(count))]


def scan_overlaps(freq: int, device: Dict[str, Any], zones: List[Dict[str, Any]]) -> bool:
    low = freq - device["half_bandwidth"]
    high = freq + device["half_bandwidth"]
    return any(high > zone["start"] and low < zone["end"] for zone in zones)


def scan_distance(freq: int, zone: Dict[str, Any]) -> int:
    if freq < zone["start"]:
        return zone["start"] - freq
    if freq > zone["end"]:
        return freq - zone["end"]
    return 0


def adjacent_required(a: Dict[str, Any], b: Dict[str, Any], rules: Dict[str, Any]) -> int:
    return a["half_bandwidth"] + b["half_bandwidth"] + rules["guard_spacing"]


# ---------------------------------------------------------------------------
# Intermodulation products
# ---------------------------------------------------------------------------


def im_products_for_source_sets(
    devices: List[Dict[str, Any]],
    source_sets: List[Tuple[int, ...]],
    rules: Dict[str, Any],
    victim_bands: Optional[List[Tuple[int, int]]] = None,
) -> List[Dict[str, Any]]:
    """Generate conservative second- and third-order positive-frequency products."""
    by_index = {d["index"]: d for d in devices}
    coupling = rules["coupling_loss_db"]
    im2 = rules["im2_rejection_db"]
    im3 = rules["im3_rejection_db"]
    products: List[Dict[str, Any]] = []

    def relevant(freq: int) -> bool:
        return victim_bands is None or any(low <= freq <= high for low, high in victim_bands)

    def add(kind: str, source_ids: Tuple[int, ...], freq: int, level: float) -> None:
        if freq > 0 and relevant(freq):
            products.append(
                {
                    "kind": kind,
                    "source_ids": list(source_ids),
                    "frequency": int(freq),
                    "level_dbm": round(float(level), 3),
                }
            )

    for source_set in source_sets:
        n = len(source_set)
        if n == 1:
            a = by_index[source_set[0]]
            # Second harmonic is retained as a conservative second-order product.
            add("IM2:2f", source_set, 2 * _freq(a), _power(a) - coupling - im2)
        elif n == 2:
            ai, bi = source_set
            a, b = by_index[ai], by_index[bi]
            fa, fb = _freq(a), _freq(b)
            pa, pb = _power(a), _power(b)

            add("IM2:f1+f2", source_set, fa + fb, pa + pb - coupling - im2)
            add("IM2:|f1-f2|", source_set, abs(fa - fb), pa + pb - coupling - im2)

            add("IM3:2f1-f2", source_set, 2 * fa - fb, 2 * pa + pb - 2 * coupling - im3)
            add("IM3:2f1+f2", source_set, 2 * fa + fb, 2 * pa + pb - 2 * coupling - im3)
            add("IM3:2f2-f1", source_set, 2 * fb - fa, 2 * pb + pa - 2 * coupling - im3)
            add("IM3:2f2+f1", source_set, 2 * fb + fa, 2 * pb + pa - 2 * coupling - im3)
        elif n == 3:
            powers = []
            freqs = []
            for idx in source_set:
                d = by_index[idx]
                freqs.append(_freq(d))
                powers.append(_power(d))
            f1, f2, f3 = freqs
            p1, p2, p3 = powers
            base_level = p1 + p2 + p3 - 2 * coupling - im3
            add("IM3:f1+f2+f3", source_set, f1 + f2 + f3, base_level)
            add("IM3:f1+f2-f3", source_set, f1 + f2 - f3, base_level)
            add("IM3:f1+f3-f2", source_set, f1 + f3 - f2, base_level)
            add("IM3:f2+f3-f1", source_set, f2 + f3 - f1, base_level)
    return products


def _freq(device: Dict[str, Any]) -> int:
    # During reporting/search every source has a selected frequency.
    return int(device["selected_frequency"])


def _power(device: Dict[str, Any]) -> float:
    return float(device["power_dbm"])


def all_source_sets(indices: List[int]) -> List[Tuple[int, ...]]:
    sets: List[Tuple[int, ...]] = []
    if indices:
            sets.extend([(i,) for i in indices])
    if len(indices) >= 2:
        sets.extend(itertools.combinations(indices, 2))
    if len(indices) >= 3:
        sets.extend(itertools.combinations(indices, 3))
    return sets

def product_threat(
    product: Dict[str, Any],
    victim: Dict[str, Any],
    rules: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    victim_idx = victim["index"]
    if victim_idx in product["source_ids"]:
        return None
    freq = product["frequency"]
    if freq < victim["band_min"] or freq > victim["band_max"]:
        return None

    separation = abs(freq - victim["selected_frequency"])
    frequency_margin = separation - victim["half_bandwidth"]
    victim_level = product["level_dbm"] - rules["coupling_loss_db"]
    level_margin = rules["victim_threshold_dbm"] - victim_level
    is_conflict = frequency_margin < 0 and level_margin < 0

    return {
        "source_type": "intermodulation_product",
        "kind": product["kind"],
        "source_device_indices": product["source_ids"],
        "frequency_mhz": hz_to_mhz(freq),
        "separation_hz": separation,
        "frequency_margin_hz": frequency_margin,
        "frequency_margin_mhz": hz_to_mhz(frequency_margin),
        "product_level_dbm": round(product["level_dbm"], 3),
        "victim_level_dbm": round(victim_level, 3),
        "level_margin_db": round(level_margin, 3),
        "is_conflict": is_conflict,
    }


# ---------------------------------------------------------------------------
# Validation and search
# ---------------------------------------------------------------------------


def prepare_inputs(req: CoordinationRequest, rule: Dict[str, Any]):
    devices = normalize_devices(req.devices)
    zones = normalize_scan_zones(req.scan_zones)
    rules = normalize_rules(rule)
    errors: List[Dict[str, Any]] = []

    all_grid: List[List[int]] = []
    domains: List[List[int]] = []

    for d in devices:
        points = grid_points(d)
        all_grid.append(points)
        safe_points = [f for f in points if not scan_overlaps(f, d, zones)]
        domains.append(safe_points)

        if not points:
            errors.append(
                {
                    "device_index": d["index"],
                    "device_name": d["name"],
                    "code": "BAND_NO_CANDIDATES",
                    "constraint": "band_min_mhz + integer channel steps must reach band_max_mhz",
                    "band_min_mhz": hz_to_mhz(d["band_min"]),
                    "band_max_mhz": hz_to_mhz(d["band_max"]),
                    "step_mhz": hz_to_mhz(d["step"]),
                }
            )
        elif not safe_points:
            errors.append(
                {
                    "device_index": d["index"],
                    "device_name": d["name"],
                    "code": "BAND_BLOCKED_BY_SCAN",
                    "constraint": "at least one grid channel must not overlap a scanned occupied zone",
                    "blocking_scan_zones": [z["name"] for z in zones],
                }
            )

        if d["locked"] is not None:
            lock_errors = validate_lock(d, points, safe_points, zones)
            errors.extend(lock_errors)

    errors.extend(validate_locked_pairs(devices, rules))

    if errors:
        raise APIError(
            409,
            "CONSTRAINT_CONFLICT",
            "No coordination plan can be created from the supplied constraints.",
            errors,
        )

    virtual_sources = []
    for zone in zones:
        for suffix, frequency in (("start", zone["start"]), ("end", zone["end"])):
            virtual_sources.append(
                {
                    "index": -(2 * zone["index"] + (1 if suffix == "end" else 0)) - 1,
                    "name": f"scan:{zone['name']}:{suffix}",
                    "kind": "scan_transmitter",
                    "band_min": 0,
                    "band_max": 100_000 * MHZ_TO_HZ,
                    "selected_frequency": frequency,
                    "half_bandwidth": 0,
                    "priority": 0,
                    "power_dbm": zone["power_dbm"],
                    "virtual": True,
                    "scan_zone_index": zone["index"],
                }
            )

    return devices, zones, rules, all_grid, domains, virtual_sources


def validate_lock(
    device: Dict[str, Any],
    grid: List[int],
    safe_points: List[int],
    zones: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    lock = int(device["locked"])
    errors = []
    in_band = device["band_min"] <= lock <= device["band_max"]
    on_grid = in_band and (lock - device["band_min"]) % device["step"] == 0

    if not in_band or not on_grid:
        errors.append(
            {
                "device_index": device["index"],
                "device_name": device["name"],
                "code": "LOCK_NOT_TUNABLE",
                "constraint": "locked frequency must be on the device step grid inside its tunable band",
                "locked_mhz": hz_to_mhz(lock),
                "band_min_mhz": hz_to_mhz(device["band_min"]),
                "band_max_mhz": hz_to_mhz(device["band_max"]),
                "step_mhz": hz_to_mhz(device["step"]),
            }
        )
    if on_grid and scan_overlaps(lock, device, zones):
        blocking = [
            z["name"]
            for z in zones
            if lock + device["half_bandwidth"] > z["start"]
            and lock - device["half_bandwidth"] < z["end"]
        ]
        errors.append(
            {
                "device_index": device["index"],
                "device_name": device["name"],
                "code": "LOCK_IN_SCAN_ZONE",
                "constraint": "locked channel may not overlap a scanned occupied zone",
                "locked_mhz": hz_to_mhz(lock),
                "blocking_scan_zones": blocking,
            }
        )
    return errors


def validate_locked_pairs(devices: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    errors = []
    locked = [(i, d) for i, d in enumerate(devices) if d["locked"] is not None]
    for (ia, a), (ib, b) in itertools.combinations(locked, 2):
        separation = abs(a["locked"] - b["locked"])
        required = adjacent_required(a, b, rules)
        if separation < required:
            errors.append(
                {
                    "code": "LOCKED_ADJACENT_CONFLICT",
                    "constraint": "locked transmitters must meet adjacent-channel protection spacing",
                    "device_indices": [ia, ib],
                    "device_names": [a["name"], b["name"]],
                    "separation_hz": separation,
                    "required_separation_hz": required,
                    "separation_mhz": hz_to_mhz(separation),
                    "required_separation_mhz": hz_to_mhz(required),
                }
            )
    return errors


def compatible(
    freq: int,
    device: Dict[str, Any],
    assigned: Dict[int, int],
    devices: List[Dict[str, Any]],
    rules: Dict[str, Any],
) -> bool:
    for j, other_freq in assigned.items():
        other = devices[j]
        if abs(freq - other_freq) < adjacent_required(device, other, rules):
            return False
    return True


def evaluate_add(
    idx: int,
    freq: int,
    assigned: Dict[int, int],
    devices: List[Dict[str, Any]],
    zones: List[Dict[str, Any]],
    rules: Dict[str, Any],
    virtual_sources: Optional[List[Dict[str, Any]]] = None,
    existing_products: Optional[List[Dict[str, Any]]] = None,
    new_product_bands: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """Return conflicts and local minimum margins introduced by this assignment."""
    virtual_sources = virtual_sources or []
    device = dict(device_with_frequency(devices[idx], freq))
    old_indices = list(assigned.keys())
    virtual_by_index = {s["index"]: s for s in virtual_sources}
    source_indices = old_indices + list(virtual_by_index.keys())
    weighted = 0.0
    raw = 0
    conflicts: List[Dict[str, Any]] = []
    min_f = math.inf
    min_l = math.inf

    def consider_margin(fm: float, lm: float) -> None:
        nonlocal min_f, min_l
        min_f = min(min_f, fm)
        min_l = min(min_l, lm)

    # Scanned external transmitters: domains already exclude overlap, but level
    # and edge distance remain part of the reported safety margin.
    for zone in zones:
        distance = scan_distance(freq, zone)
        fm = distance - device["half_bandwidth"]
        victim_level = zone["power_dbm"] - rules["coupling_loss_db"]
        lm = rules["victim_threshold_dbm"] - victim_level
        consider_margin(fm, lm)

    # Adjacent transmitters, evaluated in both receive directions.
    for j in old_indices:
        other = device_with_frequency(devices[j], assigned[j])
        separation = abs(freq - assigned[j])
        required = adjacent_required(device, other, rules)
        fm = separation - required
        victim_level_new = other["power_dbm"] - rules["coupling_loss_db"] - rules["adjacent_isolation_db"]
        victim_level_old = device["power_dbm"] - rules["coupling_loss_db"] - rules["adjacent_isolation_db"]
        consider_margin(fm, rules["victim_threshold_dbm"] - victim_level_new)
        consider_margin(fm, rules["victim_threshold_dbm"] - victim_level_old)

    # Products that already existed before this assignment can land on the new
    # victim.  Products whose source set contains the new transmitter can affect
    # either it or an older victim.  Splitting them this way counts every
    # product/victim pair exactly once.
    selected_old = {i: device_with_frequency(devices[i], assigned[i]) for i in old_indices}
    selected_old.update(virtual_by_index)
    selected = dict(selected_old)
    selected[idx] = device

    victim_bands = [(devices[j]["band_min"], devices[j]["band_max"]) for j in [idx] + old_indices]

    if existing_products is not None:
        old_products = existing_products
    else:
        old_products = im_products_for_source_sets(
            list(selected_old.values()),
            all_source_sets(source_indices),
            rules,
            victim_bands,
        )
    new_source_sets: List[Tuple[int, ...]] = [(idx,)]
    new_source_sets.extend((idx, j) for j in source_indices)
    new_source_sets.extend((idx, a, b) for a, b in itertools.combinations(source_indices, 2))
    effective_victim_bands = new_product_bands or victim_bands
    new_products = im_products_for_source_sets(
        list(selected.values()), new_source_sets, rules, effective_victim_bands
    )

    def inspect(product: Dict[str, Any], victim_idx: int, victim: Dict[str, Any]) -> None:
        nonlocal raw, weighted, min_f, min_l
        if victim_idx in product["source_ids"]:
            return
        threat = product_threat(product, victim, rules)
        if threat is None:
            return
        min_f = min(min_f, threat["frequency_margin_hz"])
        min_l = min(min_l, threat["level_margin_db"])
        if threat["is_conflict"]:
            raw += 1
            weighted += 2 ** (victim["priority"] - 1)
            threat["source_device_names"] = [
                selected[s]["name"] for s in threat["source_device_indices"]
            ]
            conflicts.append({"victim_device_index": victim_idx, **threat})

    for product in old_products:
        inspect(product, idx, device)
    for product in new_products:
        device_victims: Dict[int, Dict[str, Any]] = {idx: device}
        device_victims.update((j, selected_old[j]) for j in old_indices)
        for victim_idx, victim in device_victims.items():
            inspect(product, victim_idx, victim)

    return {
        "weighted_conflicts": weighted,
        "conflicts": raw,
        "conflict_details": conflicts,
        "min_frequency_margin_hz": int(min_f) if math.isfinite(min_f) else math.inf,
        "min_level_margin_db": round(min_l, 3) if math.isfinite(min_l) else math.inf,
        "new_products": new_products,
    }


def device_with_frequency(device: Dict[str, Any], freq: int) -> Dict[str, Any]:
    d = dict(device)
    d["selected_frequency"] = int(freq)
    return d


def solve(
    devices: List[Dict[str, Any]],
    zones: List[Dict[str, Any]],
    rules: Dict[str, Any],
    domains: List[List[int]],
    max_nodes: int,
    virtual_sources: List[Dict[str, Any]],
) -> Dict[int, int]:
    # Locks are represented as one-value domains.  The search variable ordering
    # then assigns them first without special-casing their margins or products.
    assigned: Dict[int, int] = {}
    search_domains = [
        [d["locked"]] if d["locked"] is not None else list(domains[i])
        for i, d in enumerate(devices)
    ]
    remaining = list(range(len(devices)))

    failure = {"detail": None}
    nodes = 0
    best_conflict_key = math.inf

    def record_failure(idx: int, current: Dict[int, int]) -> None:
        blockers: List[Dict[str, Any]] = []
        if current:
            for f in search_domains[idx]:
                for j in current:
                    if abs(f - current[j]) < adjacent_required(devices[idx], devices[j], rules):
                        blockers.append((f, j))
                        break
        blocked_by_indices = sorted({j for _, j in blockers})
        blocked_by = [
            {
                "device_index": j,
                "device_name": devices[j]["name"],
                "constraint": "adjacent_channel_protection",
            }
            for j in blocked_by_indices
        ]
        failure["detail"] = {
            "device_index": idx,
            "device_name": devices[idx]["name"],
            "code": "NO_FEASIBLE_FREQUENCY",
            "constraint": "no remaining candidate satisfies adjacent-channel protection",
            "blocked_by_devices": blocked_by,
            "blocked_candidate_count": len({f for f, _ in blockers}),
            "candidate_count_after_scan": len(search_domains[idx]),
        }

    def select_variable(current: Dict[int, int]) -> Optional[int]:
        options = []
        for i in remaining:
            if i in current:
                continue
            count = sum(1 for f in search_domains[i] if compatible(f, devices[i], current, devices, rules))
            if count == 0:
                record_failure(i, current)
                return None
            options.append((count, -devices[i]["priority"], i))
        return min(options)[2] if options else None

    def candidate_key(i: int, f: int, current: Dict[int, int], existing_products):
        ev = evaluate_add(
            i, f, current, devices, zones, rules, virtual_sources, existing_products
        )
        return (
            ev["weighted_conflicts"],
            ev["conflicts"],
            -ev["min_level_margin_db"],
            -ev["min_frequency_margin_hz"],
            f,
        )

    def dfs(
        current: Dict[int, int],
        weighted: float,
        raw: int,
        min_f: int,
        min_l: float,
        products: List[Dict[str, Any]],
        best_obj: Tuple[float, int, int, float],
    ) -> Optional[Tuple[Dict[int, int], Tuple[float, int, int, float]]]:
        nonlocal nodes, best_conflict_key
        nodes += 1
        if nodes > max_nodes:
            raise APIError(
                409,
                "SEARCH_LIMIT_REACHED",
                f"Search exceeded {max_nodes} nodes. Tighten scan zones, lock a frequency, or raise max_search_nodes.",
            )

        if len(current) == len(devices):
            return dict(current), (weighted, raw, -min_f, -min_l)

        if weighted >= best_conflict_key:
            return None

        var = select_variable(current)
        if var is None:
            return None

        candidates = [f for f in search_domains[var] if compatible(f, devices[var], current, devices, rules)]
        scored = []
        evaluations: Dict[int, Dict[str, Any]] = {}
        for f in candidates:
            ev = evaluate_add(
                var,
                f,
                current,
                devices,
                zones,
                rules,
                virtual_sources,
                products,
            )
            evaluations[f] = ev
            scored.append(
                (
                    ev["weighted_conflicts"],
                    ev["conflicts"],
                    -ev["min_level_margin_db"],
                    -ev["min_frequency_margin_hz"],
                    f,
                )
            )
        scored.sort()

        best = None
        for _, _, _, _, f in scored:
            ev = evaluations[f]
            new_weighted = weighted + ev["weighted_conflicts"]
            new_raw = raw + ev["conflicts"]
            new_min_f = min(min_f, ev["min_frequency_margin_hz"])
            new_min_l = min(min_l, ev["min_level_margin_db"])
            tentative_obj = (new_weighted, new_raw, -new_min_f, -new_min_l)
            if new_weighted >= best_conflict_key:
                continue
            if tentative_obj >= best_obj:
                continue

            current[var] = f
            outcome = dfs(
                current,
                new_weighted,
                new_raw,
                new_min_f,
                new_min_l,
                products + ev["new_products"],
                best_obj,
            )
            del current[var]
            if outcome is not None and (best is None or outcome[1] < best[1]):
                best = outcome
                best_obj = outcome[1]
                best_conflict_key = outcome[1][0]
                if best_conflict_key == 0:
                    return best
        return best

    virtual_indices = [s["index"] for s in virtual_sources]
    device_bands = [(d["band_min"], d["band_max"]) for d in devices]
    initial_products = im_products_for_source_sets(
        virtual_sources,
        all_source_sets(virtual_indices),
        rules,
        device_bands,
    )

    # Greedy first solution gives branch-and-bound an early upper bound.
    greedy: Dict[int, int] = {}
    greedy_products = list(initial_products)
    greedy_state = (0.0, 0, math.inf, math.inf)
    greedy_ok = True
    while len(greedy) < len(devices):
        var = select_variable(greedy)
        if var is None:
            greedy_ok = False
            break
        candidates = [f for f in search_domains[var] if compatible(f, devices[var], greedy, devices, rules)]
        f = min(candidates, key=lambda x: candidate_key(var, x, greedy, greedy_products))
        ev = evaluate_add(
            var,
            f,
            greedy,
            devices,
            zones,
            rules,
            virtual_sources,
            greedy_products,
        )
        greedy[var] = f
        greedy_products.extend(ev["new_products"])
        greedy_state = (
            greedy_state[0] + ev["weighted_conflicts"],
            greedy_state[1] + ev["conflicts"],
            min(greedy_state[2], ev["min_frequency_margin_hz"]),
            min(greedy_state[3], ev["min_level_margin_db"]),
        )

    best_obj = (math.inf, math.inf, math.inf, math.inf)
    if greedy_ok:
        best_conflict_key = greedy_state[0]
        best_obj = (
            greedy_state[0],
            greedy_state[1],
            -greedy_state[2],
            -greedy_state[3],
        )

    result = dfs({}, 0.0, 0, math.inf, math.inf, initial_products, best_obj)
    if result is not None:
        return result[0]

    if greedy_ok:
        # DFS only accepts a strictly better tie-break than the greedy upper
        # bound, so an equal valid greedy solution is deliberately returned here.
        return greedy

    detail = failure["detail"] or {
        "code": "NO_FEASIBLE_SOLUTION",
        "constraint": "adjacent-channel and scan-zone constraints",
    }
    raise APIError(
        409,
        "NO_FEASIBLE_SOLUTION",
        "No feasible frequency assignment exists under the supplied constraints.",
        [detail],
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def build_report(
    req: CoordinationRequest,
    rule: Dict[str, Any],
    devices: List[Dict[str, Any]],
    zones: List[Dict[str, Any]],
    rules: Dict[str, Any],
    all_grid: List[List[int]],
    domains: List[List[int]],
    assignment: Dict[int, int],
    virtual_sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    selected = [device_with_frequency(d, assignment[i]) for i, d in enumerate(devices)]
    indices = [d["index"] for d in selected]
    source_devices = selected + virtual_sources
    source_by_id = {s["index"]: s for s in source_devices}
    source_indices = indices + [s["index"] for s in virtual_sources]
    all_products = im_products_for_source_sets(
        source_devices,
        all_source_sets(source_indices),
        rules,
        [(d["band_min"], d["band_max"]) for d in selected],
    )
    # Products not sourced by a given planned device.  They are the baseline
    # used when ranking alternative frequencies for that device.
    products_without_device: Dict[int, List[Dict[str, Any]]] = {}
    for d in selected:
        products_without_device[d["index"]] = [
            p for p in all_products if d["index"] not in p["source_ids"]
        ]

    device_reports = []
    total_conflicts = 0
    weighted_conflicts = 0.0
    global_min_f = math.inf
    global_min_l = math.inf

    for victim in selected:
        idx = victim["index"]
        freq = assignment[idx]

        scan_threats = []
        for zone in zones:
            distance = scan_distance(freq, zone)
            fm = distance - victim["half_bandwidth"]
            victim_level = zone["power_dbm"] - rules["coupling_loss_db"]
            lm = rules["victim_threshold_dbm"] - victim_level
            threat = {
                "source_type": "scan_zone",
                "source_name": zone["name"],
                "frequency_mhz": hz_to_mhz(zone["start"] if freq < zone["start"] else zone["end"]),
                "separation_hz": distance,
                "frequency_margin_hz": int(fm),
                "frequency_margin_mhz": hz_to_mhz(fm),
                "victim_level_dbm": round(victim_level, 3),
                "level_margin_db": round(lm, 3),
                "is_conflict": False,  # overlapping scan channels were hard-filtered
            }
            scan_threats.append(threat)
            global_min_f = min(global_min_f, fm)
            global_min_l = min(global_min_l, lm)

        adjacent_threats = []
        for other in selected:
            if other["index"] == idx:
                continue
            separation = abs(freq - assignment[other["index"]])
            required = adjacent_required(victim, other, rules)
            fm = separation - required
            victim_level = other["power_dbm"] - rules["coupling_loss_db"] - rules["adjacent_isolation_db"]
            lm = rules["victim_threshold_dbm"] - victim_level
            adjacent_threats.append(
                {
                    "source_type": "transmitter",
                    "source_device_index": other["index"],
                    "source_name": other["name"],
                    "source_kind": other["kind"],
                    "source_frequency_mhz": hz_to_mhz(assignment[other["index"]]),
                    "separation_hz": separation,
                    "required_separation_hz": required,
                    "frequency_margin_hz": int(fm),
                    "frequency_margin_mhz": hz_to_mhz(fm),
                    "victim_level_dbm": round(victim_level, 3),
                    "level_margin_db": round(lm, 3),
                    "is_conflict": False,  # minimum guard is a hard feasibility constraint
                }
            )
            global_min_f = min(global_min_f, fm)
            global_min_l = min(global_min_l, lm)

        im_threats = []
        for product in all_products:
            threat = product_threat(product, victim, rules)
            if threat is not None:
                threat["source_device_names"] = [
                    source_by_id[s]["name"] for s in threat["source_device_indices"]
                ]
                im_threats.append(threat)
                global_min_f = min(global_min_f, threat["frequency_margin_hz"])
                global_min_l = min(global_min_l, threat["level_margin_db"])
                if threat["is_conflict"]:
                    total_conflicts += 1
                    weighted_conflicts += 2 ** (victim["priority"] - 1)

        im_threats.sort(key=lambda t: (t["separation_hz"], t["kind"]))
        adjacent_threats.sort(key=lambda t: t["separation_hz"])
        scan_threats.sort(key=lambda t: t["separation_hz"])
        all_threats = scan_threats + adjacent_threats + im_threats
        nearest = min(all_threats, key=lambda t: t["separation_hz"], default=None)

        alternatives = []
        other_assigned = {j: f for j, f in assignment.items() if j != idx}
        for candidate in domains[idx]:
            if not compatible(candidate, victim, other_assigned, devices, rules):
                continue
            ev = evaluate_add(
                idx,
                candidate,
                other_assigned,
                devices,
                [],
                rules,
                virtual_sources,
                products_without_device[idx],
                [(devices[idx]["band_min"], devices[idx]["band_max"])],
            )
            alternatives.append(
                (
                    ev["weighted_conflicts"],
                    ev["conflicts"],
                    -ev["min_level_margin_db"],
                    -ev["min_frequency_margin_hz"],
                    candidate,
                )
            )
        alternatives.sort()
        rank = next((r + 1 for r, alt in enumerate(alternatives) if alt[-1] == freq), None)
        adjacent_safe_count = len(alternatives)

        conflict_count = sum(1 for t in im_threats if t["is_conflict"])
        reasons = [
            f"priority {victim['priority']}",
            f"{len(domains[idx])} scan-safe grid channels",
            f"{adjacent_safe_count} channels satisfy the final adjacent-channel guard",
        ]
        if victim["locked"] is not None:
            reasons.insert(0, "frequency is locked, therefore forced")
        else:
            reasons.insert(0, f"rank {rank} by weighted conflicts and safety margin")

        device_reports.append(
            {
                "device_index": idx,
                "device_name": victim["name"],
                "device_kind": victim["kind"],
                "priority": victim["priority"],
                "assigned_frequency_mhz": hz_to_mhz(freq),
                "locked": victim["locked"] is not None,
                "tx_power_dbm": victim["power_dbm"],
                "bandwidth_mhz": hz_to_mhz(victim["bandwidth"]),
                "grid_candidate_count": len(all_grid[idx]),
                "scan_safe_candidate_count": len(domains[idx]),
                "adjacent_safe_candidate_count": adjacent_safe_count,
                "alternative_rank": rank,
                "conflict_count": conflict_count,
                "selection_basis": reasons,
                "nearest_interference_source": nearest,
                "minimum_frequency_margin_hz": min(
                    [t["frequency_margin_hz"] for t in all_threats], default=None
                ),
                "minimum_level_margin_db": round(
                    min((t["level_margin_db"] for t in all_threats), default=math.inf), 3
                )
                if all_threats
                else None,
                "scan_threats": scan_threats,
                "adjacent_threats": adjacent_threats,
                "intermodulation_threats": im_threats,
            }
        )

    if not math.isfinite(global_min_f):
        global_min_f = 0
    if not math.isfinite(global_min_l):
        global_min_l = 0.0

    result = {
        "scheme": {
            "name": req.name,
            "rule_version": {
                "id": rule["id"],
                "name": rule["name"],
            },
            "parameters": {
                "guard_spacing_mhz": rule["guard_spacing_mhz"],
                "adjacent_isolation_db": rule["adjacent_isolation_db"],
                "coupling_loss_db": rule["coupling_loss_db"],
                "im2_rejection_db": rule["im2_rejection_db"],
                "im3_rejection_db": rule["im3_rejection_db"],
                "victim_threshold_dbm": rule["victim_threshold_dbm"],
            },
            "total_conflict_count": total_conflicts,
            "weighted_conflict_count": weighted_conflicts,
            "minimum_frequency_margin_hz": int(global_min_f),
            "minimum_frequency_margin_mhz": hz_to_mhz(int(global_min_f)),
            "minimum_level_margin_db": round(global_min_l, 3),
            "search_node_limit": req.max_search_nodes,
        },
        "assignments": [
            {
                "device_index": i,
                "device_name": devices[i]["name"],
                "frequency_mhz": hz_to_mhz(assignment[i]),
                "locked": devices[i]["locked"] is not None,
            }
            for i in range(len(devices))
        ],
        "device_reports": device_reports,
    }
    return result


def coordinate(req: CoordinationRequest, rule: Dict[str, Any]) -> Dict[str, Any]:
    devices, zones, rules, all_grid, domains, virtual_sources = prepare_inputs(req, rule)
    assignment = solve(devices, zones, rules, domains, req.max_search_nodes, virtual_sources)
    return build_report(
        req, rule, devices, zones, rules, all_grid, domains, assignment, virtual_sources
    )


# ---------------------------------------------------------------------------
# Plan persistence and version operations
# ---------------------------------------------------------------------------


def save_plan(
    conn: sqlite3.Connection,
    req: CoordinationRequest,
    rule_row: sqlite3.Row,
    result: Dict[str, Any],
    plan_id: Optional[str] = None,
    parent_plan_id: Optional[str] = None,
) -> str:
    plan_id = plan_id or str(uuid.uuid4())
    scheme = result["scheme"]
    conn.execute(
        """
        INSERT INTO plans(
            id, name, rule_id, request_json, result_json, parent_plan_id, note,
            created_at, conflict_count, weighted_conflict_count,
            min_frequency_margin_hz, min_level_margin_db
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            plan_id,
            req.name,
            rule_row["id"],
            req.model_dump_json(),
            json.dumps(result, ensure_ascii=False),
            parent_plan_id,
            req.note,
            utc_now(),
            scheme["total_conflict_count"],
            scheme["weighted_conflict_count"],
            scheme["minimum_frequency_margin_hz"],
            scheme["minimum_level_margin_db"],
        ),
    )
    conn.commit()
    return plan_id


def plan_summary(row: sqlite3.Row) -> Dict[str, Any]:
    result = json.loads(row["result_json"])
    return {
        "id": row["id"],
        "name": row["name"],
        "rule_id": row["rule_id"],
        "parent_plan_id": row["parent_plan_id"],
        "note": row["note"],
        "created_at": row["created_at"],
        "conflict_count": row["conflict_count"],
        "weighted_conflict_count": row["weighted_conflict_count"],
        "minimum_frequency_margin_hz": row["min_frequency_margin_hz"],
        "minimum_frequency_margin_mhz": hz_to_mhz(row["min_frequency_margin_hz"]),
        "minimum_level_margin_db": row["min_level_margin_db"],
        "assignments": result["assignments"],
    }


def full_plan(row: sqlite3.Row, rule_row: Optional[sqlite3.Row] = None) -> Dict[str, Any]:
    data = {
        "id": row["id"],
        "name": row["name"],
        "note": row["note"],
        "parent_plan_id": row["parent_plan_id"],
        "created_at": row["created_at"],
        "request": json.loads(row["request_json"]),
        "result": json.loads(row["result_json"]),
    }
    if rule_row is not None:
        data["rule_version"] = rule_row_to_dict(rule_row)
    return data


@app.get("/health")
def health(request: Request):
    db = get_db(request)
    db.execute("SELECT 1")
    return {"status": "ok"}


@app.post("/api/rules", status_code=201)
def create_rule(rule: RuleInput, request: Request):
    db = get_db(request)
    try:
        cur = db.execute(
            "INSERT INTO rule_versions(name, rules_json, active, created_at) VALUES (?, ?, 1, ?)",
            (rule.name, rule.model_dump_json(), utc_now()),
        )
        db.execute("UPDATE rule_versions SET active = 0 WHERE id <> ?", (cur.lastrowid,))
        db.commit()
    except sqlite3.IntegrityError:
        raise APIError(409, "DUPLICATE_RULE_NAME", f"Rule name {rule.name} already exists.")
    row = db.execute("SELECT * FROM rule_versions WHERE id = ?", (cur.lastrowid,)).fetchone()
    return rule_row_to_dict(row)


@app.get("/api/rules")
def list_rules(request: Request):
    rows = get_db(request).execute("SELECT * FROM rule_versions ORDER BY id DESC").fetchall()
    return {"rules": [rule_row_to_dict(r) for r in rows]}


@app.get("/api/rules/{rule_id}")
def get_rule(rule_id: int, request: Request):
    row = get_db(request).execute("SELECT * FROM rule_versions WHERE id = ?", (rule_id,)).fetchone()
    if row is None:
        raise APIError(404, "RULE_NOT_FOUND", f"Rule version {rule_id} does not exist.")
    return rule_row_to_dict(row)


@app.post("/api/coordination/plans", status_code=201)
def create_plan(req: CoordinationRequest, request: Request):
    db = get_db(request)
    rule_row = fetch_rule(db, req.rule_id)
    rule = rule_row_to_dict(rule_row)
    result = coordinate(req, rule)
    plan_id = save_plan(db, req, rule_row, result)
    return {"plan_id": plan_id, **result}


@app.get("/api/coordination/plans")
def list_plans(request: Request, limit: int = 50, offset: int = 0):
    limit = min(max(limit, 1), 200)
    rows = (
        get_db(request)
        .execute(
            "SELECT * FROM plans ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
            (limit, max(offset, 0)),
        )
        .fetchall()
    )
    return {"plans": [plan_summary(r) for r in rows]}


@app.get("/api/coordination/plans/{plan_id}")
def get_plan(plan_id: str, request: Request):
    db = get_db(request)
    row = db.execute("SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone()
    if row is None:
        raise APIError(404, "PLAN_NOT_FOUND", f"Plan {plan_id} does not exist.")
    rule_row = db.execute("SELECT * FROM rule_versions WHERE id = ?", (row["rule_id"],)).fetchone()
    return full_plan(row, rule_row)


@app.get("/api/coordination/plans/{plan_id}/export")
def export_plan(plan_id: str, request: Request):
    return get_plan(plan_id, request)


@app.post("/api/coordination/plans/{plan_id}/recalculate", status_code=201)
def recalculate_plan(plan_id: str, req: RecalculateRequest, request: Request):
    db = get_db(request)
    old_row = db.execute("SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone()
    if old_row is None:
        raise APIError(404, "PLAN_NOT_FOUND", f"Plan {plan_id} does not exist.")

    old_request = json.loads(old_row["request_json"])
    devices_data = old_request.get("devices", [])

    for idx, replacement in req.replacements.items():
        if idx < 0 or idx >= len(devices_data):
            raise APIError(
                404,
                "DEVICE_NOT_FOUND",
                f"Device index {idx} does not exist in the old plan.",
                [{"device_index": idx, "device_count": len(devices_data)}],
            )
        devices_data[idx] = replacement.model_dump()

    if req.locked_mhz or req.unlocked_indices:
        for idx in set(list(req.locked_mhz.keys()) + req.unlocked_indices):
            if idx < 0 or idx >= len(devices_data):
                raise APIError(
                    404,
                    "DEVICE_NOT_FOUND",
                    f"Device index {idx} does not exist in the old plan.",
                    [{"device_index": idx, "device_count": len(devices_data)}],
                )
        for idx, value in req.locked_mhz.items():
            devices_data[idx]["locked_mhz"] = value
        for idx in req.unlocked_indices:
            devices_data[idx]["locked_mhz"] = None

    scan_zones = old_request.get("scan_zones", [])
    scan_zones.extend(z.model_dump() for z in req.additional_scan_zones)

    new_request_data = {
        "name": req.name or f"{old_row['name']} (revised)",
        "devices": devices_data,
        "scan_zones": scan_zones,
        "rule_id": req.rule_id if req.rule_id is not None else old_row["rule_id"],
        "note": req.note,
        "max_search_nodes": req.max_search_nodes,
    }
    try:
        new_req = CoordinationRequest(**new_request_data)
    except ValidationError as exc:
        # Reuse the same structured error shape as direct plan creation.
        return validation_error_response(exc)

    rule_row = fetch_rule(db, new_req.rule_id)
    rule = rule_row_to_dict(rule_row)
    result = coordinate(new_req, rule)
    new_id = save_plan(db, new_req, rule_row, result, parent_plan_id=plan_id)
    return {"plan_id": new_id, "parent_plan_id": plan_id, **result}


@app.post("/api/coordination/compare")
def compare_plans(req: CompareRequest, request: Request):
    db = get_db(request)

    def load(plan_id: str) -> sqlite3.Row:
        row = db.execute("SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone()
        if row is None:
            raise APIError(404, "PLAN_NOT_FOUND", f"Plan {plan_id} does not exist.")
        return row

    row_a, row_b = load(req.plan_id_a), load(req.plan_id_b)
    a = json.loads(row_a["result_json"])
    b = json.loads(row_b["result_json"])
    sa, sb = a["scheme"], b["scheme"]

    freq_a = {x["device_index"]: x["frequency_mhz"] for x in a["assignments"]}
    freq_b = {x["device_index"]: x["frequency_mhz"] for x in b["assignments"]}
    changed = []
    for idx in sorted(set(freq_a) | set(freq_b)):
        if freq_a.get(idx) != freq_b.get(idx):
            changed.append(
                {
                    "device_index": idx,
                    "frequency_a_mhz": freq_a.get(idx),
                    "frequency_b_mhz": freq_b.get(idx),
                    "delta_b_minus_a_mhz": None
                    if freq_a.get(idx) is None or freq_b.get(idx) is None
                    else round(freq_b[idx] - freq_a[idx], 6),
                }
            )

    return {
        "plan_a": plan_summary(row_a),
        "plan_b": plan_summary(row_b),
        "delta": {
            "conflict_count_b_minus_a": sb["total_conflict_count"] - sa["total_conflict_count"],
            "weighted_conflict_count_b_minus_a": round(
                sb["weighted_conflict_count"] - sa["weighted_conflict_count"], 3
            ),
            "minimum_frequency_margin_b_minus_a_hz": sb[
                "minimum_frequency_margin_hz"
            ]
            - sa["minimum_frequency_margin_hz"],
            "minimum_level_margin_b_minus_a_db": round(
                sb["minimum_level_margin_db"] - sa["minimum_level_margin_db"], 3
            ),
            "changed_frequency_count": len(changed),
            "changed_frequencies": changed,
        },
    }
