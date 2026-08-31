import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from ddareungi_rearrangement.latent_sensitivity import (
    build_latent_demand_sensitivity,
    run_latent_demand_sensitivity,
)

TRAINING_START = datetime(2025, 11, 3)
TRAINING_END = datetime(2025, 11, 6)
EVALUATION_START = datetime(2025, 11, 10)
EVALUATION_END = datetime(2025, 11, 10, 1)


def _station_hour() -> pl.DataFrame:
    timestamps: list[datetime] = []
    available_bikes: list[int] = []
    rentals: list[int] = []
    for day_offset in range(3):
        start = TRAINING_START + timedelta(days=day_offset)
        timestamps.extend([start, start + timedelta(hours=1)])
        available_bikes.extend([5, 5])
        rentals.extend([5, 0])
    timestamps.extend([EVALUATION_START, EVALUATION_END])
    available_bikes.extend([0, 0])
    rentals.extend([0, 0])
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "station_id": ["A"] * len(timestamps),
            "station_name": ["에이"] * len(timestamps),
            "available_bikes": available_bikes,
            "rentals": rentals,
            "inventory_observed": [True] * len(timestamps),
            "actionable": [True] * len(timestamps),
        }
    )


def _trips() -> pl.DataFrame:
    rows = []
    for index in range(20):
        rent_at = TRAINING_START + timedelta(days=index % 3, minutes=index)
        rows.append(
            {
                "rent_at": rent_at,
                "return_at": rent_at + timedelta(minutes=10 + index),
                "rent_station_id": "A",
                "return_station_id": "OUT",
                "duration_minutes": 10 + index,
                "rent_in_scope": True,
            }
        )
    rows.append(
        {
            "rent_at": EVALUATION_START + timedelta(minutes=50),
            "return_at": EVALUATION_END + timedelta(minutes=20),
            "rent_station_id": "A",
            "return_station_id": "OUT",
            "duration_minutes": 30,
            "rent_in_scope": True,
        }
    )
    return pl.DataFrame(rows)


def test_small_paired_batch_reuses_hashes_and_preserves_invariants() -> None:
    results = run_latent_demand_sensitivity(
        _trips(),
        _station_hour(),
        {"A": (37.5, 127.0)},
        training_start=TRAINING_START,
        training_end=TRAINING_END,
        evaluation_start=EVALUATION_START,
        evaluation_end=EVALUATION_END,
        seeds=(20251124, 20251125),
    )

    assert results.policy_runs.height == 12
    assert results.paired_runs.height == 6
    assert results.scenario_summary.height == 3
    assert (
        results.policy_runs.group_by("seed", "scenario")
        .agg(
            pl.col("manifest_hash").n_unique().alias("hashes"),
            pl.col("combined_requests").n_unique().alias("request_counts"),
        )["hashes"]
        .to_list()
        == [1] * 6
    )
    assert (
        results.policy_runs.group_by("seed", "scenario")
        .agg(pl.col("combined_requests").n_unique().alias("request_counts"))["request_counts"]
        .to_list()
        == [1] * 6
    )
    assert results.policy_runs["total_conservation_residual"].to_list() == [0] * 12
    assert results.policy_runs["combined_trip_flow_residual"].to_list() == [0] * 12


def test_build_small_batch_writes_reproducible_artifacts(tmp_path: Path) -> None:
    trips_path = tmp_path / "trips.parquet"
    station_hour_path = tmp_path / "station_hour.parquet"
    coordinate_path = tmp_path / "coordinates.csv"
    _trips().write_parquet(trips_path)
    _station_hour().write_parquet(station_hour_path)
    pl.DataFrame({"station_id": ["A"], "latitude": [37.5], "longitude": [127.0]}).write_csv(
        coordinate_path
    )
    outputs = {
        "policy": tmp_path / "policy.csv",
        "paired": tmp_path / "paired.csv",
        "summary": tmp_path / "summary.csv",
        "figure": tmp_path / "figure.png",
        "json": tmp_path / "report.json",
        "markdown": tmp_path / "report.md",
    }

    experiment = build_latent_demand_sensitivity(
        trips_path=trips_path,
        station_hour_path=station_hour_path,
        coordinate_path=coordinate_path,
        training_start=TRAINING_START,
        training_end=TRAINING_END,
        evaluation_start=EVALUATION_START,
        evaluation_end=EVALUATION_END,
        policy_runs_csv_path=outputs["policy"],
        paired_runs_csv_path=outputs["paired"],
        summary_csv_path=outputs["summary"],
        figure_path=outputs["figure"],
        json_path=outputs["json"],
        markdown_path=outputs["markdown"],
        seeds=(20251124, 20251125),
        expected_observed_contract=None,
        enforce_frozen_seed_set=False,
    )

    assert experiment.seeds == (20251124, 20251125)
    assert pl.read_csv(outputs["policy"]).height == 12
    assert pl.read_csv(outputs["paired"]).height == 6
    assert pl.read_csv(outputs["summary"]).height == 3
    assert outputs["figure"].stat().st_size > 0
    report = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert report["contract_version"] == "latent-demand-v2"
    assert report["stop_required"] is True
    assert "STOP_REQUIRED" in outputs["markdown"].read_text(encoding="utf-8")


def test_final_batch_rejects_non_frozen_seed_set() -> None:
    with pytest.raises(ValueError, match="동결된 seed"):
        run_latent_demand_sensitivity(
            _trips(),
            _station_hour(),
            {"A": (37.5, 127.0)},
            training_start=TRAINING_START,
            training_end=TRAINING_END,
            evaluation_start=EVALUATION_START,
            evaluation_end=EVALUATION_END,
            seeds=(20251124,),
            enforce_frozen_seed_set=True,
        )
