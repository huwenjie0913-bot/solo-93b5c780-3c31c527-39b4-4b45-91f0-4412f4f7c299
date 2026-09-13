# Live RF Coordination Service

FastAPI + sqlite3 service for coordinating wireless microphones, in-ear monitors,
and intercom transmitters. It treats scan-zone overlap and adjacent-channel guard
as hard constraints, then minimizes second-/third-order intermodulation conflicts.

## Run

```bash
python3 -m pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

The SQLite database defaults to `./rf_coordination.db`; override it with
`RF_DB_PATH=/path/to/rf.db`.

## Frequency model

* All request/response frequencies are MHz, rounded to integer Hz internally.
* Candidate channels begin at `band_min_mhz` and advance by `step_mhz` while not
  exceeding `band_max_mhz`.
* A hard blocked candidate has any overlap with a scanned occupied zone:
  `|channel - zone| < half channel bandwidth`.
* A hard adjacent violation occurs when two selected channels are closer than
  `half_bw_a + half_bw_b + guard_spacing`.
* IM products are scored as soft conflicts. A product is a conflict when it is
  inside the victim tunable band, inside the victim channel-edge margin, and its
  estimated delivered level exceeds `victim_threshold_dbm`.

Conservative products include second harmonic and two-tone IM2
(`2f`, `f1+f2`, `|f1-f2|`) and IM3 (`2f1±f2`, `2f2±f1`, three-source sum/difference
forms). Scan zones are also represented as virtual external transmitters so
external-only and mixed intermodulation products can land on planned receivers.

## Main endpoints

| Method and path | Purpose |
| --- | --- |
| `POST /api/coordination/plans` | Calculate and persist a plan. |
| `GET /api/coordination/plans` | Browse historical plan summaries. |
| `GET /api/coordination/plans/{id}` | Fetch a stored plan and full per-device report. |
| `GET /api/coordination/plans/{id}/export` | JSON export (same full JSON body). |
| `POST /api/coordination/plans/{id}/recalculate` | Replace devices, add tightened scan zones, or lock/unlock channels. |
| `POST /api/coordination/compare` | Compare two plans' conflict counts and minimum margins. |
| `GET/POST /api/rules` | List or create rule versions. New rule versions become active. |

Interactive docs are available at `/docs` when running Uvicorn.

## Example request

```json
{
  "name": "main stage",
  "devices": [
    {
      "name": "vocal mic 1",
      "kind": "mic",
      "band_min_mhz": 470.0,
      "band_max_mhz": 490.0,
      "step_mhz": 0.025,
      "bandwidth_mhz": 0.2,
      "priority": 5,
      "tx_power_dbm": 20,
      "locked_mhz": 475.0
    },
    {
      "name": "lead IEM",
      "kind": "iem",
      "band_min_mhz": 470.0,
      "band_max_mhz": 490.0,
      "step_mhz": 0.025,
      "bandwidth_mhz": 0.2,
      "priority": 4,
      "tx_power_dbm": 20
    },
    {
      "name": "stage intercom",
      "kind": "intercom",
      "band_min_mhz": 470.0,
      "band_max_mhz": 490.0,
      "step_mhz": 0.0125,
      "bandwidth_mhz": 0.125,
      "priority": 2,
      "tx_power_dbm": 25
    }
  ],
  "scan_zones": [
    {"name": "house TX A", "start_mhz": 481.0, "end_mhz": 481.4, "power_dbm": -25}
  ],
  "max_search_nodes": 20000
}
```

The response contains the selected frequencies, total conflict count, weighted
conflict count, and one report per device. Each report gives candidate counts,
alternative rank, selection basis, nearest interference source, minimum
frequency/level margins, scan threats, adjacent-channel threats, and all in-band
intermodulation threats.

## Error shape

Errors use:

```json
{
  "error": {
    "code": "CONSTRAINT_CONFLICT",
    "message": "...",
    "details": [
      {"device_index": 2, "device_name": "stage intercom", "code": "LOCK_NOT_TUNABLE"}
    ]
  }
}
```

The details identify the device index/name and the violated band, step, scan,
lock, adjacent-channel, or no-feasible-solution constraint.
