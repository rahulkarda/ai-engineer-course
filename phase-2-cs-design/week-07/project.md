# Week 07 Project — Parking-Lot Management System

> Deliverable: A production-quality Python package.  ·  Time: ~3 hr  ·  Due: End of Day 7

---

## Brief

Implement the **parking-lot system** designed in Day 6, incorporating all the production-quality steps from Day 7. Deliver it as a runnable Python package with full tests.

---

## Requirements

### Functional

| # | Requirement | Acceptance test |
|---|-------------|-----------------|
| F1 | Park a vehicle → receive a `Ticket` | `lot.park(vehicle)` returns `Ticket` with correct spot and entry time |
| F2 | Exit with ticket → receive fee | `lot.exit(ticket_id)` returns correct hourly fee |
| F3 | Motorcycle fits Small/Medium/Large; Car fits Medium/Large; Truck fits Large only | `SpotSize.can_fit` tests |
| F4 | `SmallestFitStrategy` parks vehicle in smallest compatible spot | Motorcycle lands in Small before Medium |
| F5 | Lot full → `ParkingLotFullError` | Raised when all compatible spots occupied |
| F6 | Duplicate plate → `DuplicatePlateError` | Second park of same plate raises |
| F7 | Unknown ticket → `UnknownTicketError` | `lot.exit("bad-id")` raises |
| F8 | Occupancy stats correct after park/exit | `lot.occupancy()` totals match operations |
| F9 | Config from environment variables (prefix `PARKING_`) | Settings loaded by `get_settings()` |
| F10 | CLI: `park`, `exit`, `status` subcommands work | End-to-end manual test |

### Non-functional

- Python 3.12+, type hints everywhere.
- `pytest` coverage ≥ 85%.
- No secrets or hardcoded rates — all from environment / defaults.
- `ruff check .` passes with no errors.
- Package installable via `pip install -e .`.

---

## Deliverable structure

```
week-07-project/
├── parking_lot/
│   ├── __init__.py
│   ├── models.py        # Vehicle, ParkingSpot, Ticket, exceptions
│   ├── fee.py           # FeeCalculator ABC + HourlyFeeCalculator
│   ├── lot.py           # Level, ParkingLot, SpotSelectionStrategy
│   ├── config.py        # ParkingLotSettings via pydantic-settings
│   └── cli.py           # argparse CLI
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_fee.py
│   └── test_lot.py
└── pyproject.toml
```

---

## Patterns applied (must be present)

Document in a comment block at the top of each relevant file:

| File | Pattern(s) |
|------|-----------|
| `lot.py` | Strategy (spot selection), Factory helper, Dependency Injection |
| `fee.py` | Strategy (fee algorithm) |
| `models.py` | — |
| `config.py` | Singleton (via `@lru_cache`) |

---

## Grading rubric

| Criterion | Points |
|-----------|--------|
| All F1–F10 functional tests pass | 40 |
| Domain exceptions hierarchy (`ParkingError` base) | 10 |
| Config loaded from env (no hardcoded values) | 10 |
| `SmallestFitStrategy` and at least one alternative strategy | 10 |
| Test coverage ≥ 85% (`pytest --cov=parking_lot`) | 15 |
| `ruff check .` clean | 5 |
| At least one stretch goal (concurrent, persistence, or async API) | 10 (bonus) |

**Pass threshold: 75 / 90 base points.**

---

## Stretch goals

1. **Thread-safety.** Add per-spot `threading.Lock`s. Write a concurrent test with `concurrent.futures.ThreadPoolExecutor`.
2. **SQLite persistence.** `SqliteStateRepository` saves lot state; reloads on startup.
3. **FastAPI wrapper.** `POST /park`, `POST /exit/{ticket_id}`, `GET /status` with proper dependency injection via `Depends`.
4. **Observability.** Observer event bus firing `VehicleParked` / `VehicleExited`; `MetricsObserver` counting events.

---

## Submission checklist

- [ ] `pip install -e ".[dev]"` succeeds.
- [ ] `pytest tests/ -v --cov=parking_lot --cov-report=term-missing` shows ≥ 85% coverage and all tests green.
- [ ] `ruff check .` exits 0.
- [ ] CLI demo: `python -m parking_lot.cli park --plate ABC123 --type car` prints a ticket ID.
- [ ] All patterns listed in the table above are present and working.
