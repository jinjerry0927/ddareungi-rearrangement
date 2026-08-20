from datetime import datetime, timedelta

import polars as pl
import pytest

from ddareungi_rearrangement.latent_demand import (
    CONTRACT_VERSION,
    LatentDemandConfig,
    LatentDemandError,
    build_latent_demand_manifest,
)

TRAINING_START = datetime(2025, 11, 3)
TRAINING_END = datetime(2025, 11, 6)
EVALUATION_START = datetime(2025, 11, 10)
EVALUATION_END = datetime(2025, 11, 10, 1)


def _config(*, seed: int = 20251124) -> LatentDemandConfig:
    return LatentDemandConfig(
        training_start=TRAINING_START,
        training_end=TRAINING_END,
        evaluation_start=EVALUATION_START,
        evaluation_end=EVALUATION_END,
        seed=seed,
    )


def _station_hour(
    *,
    evaluation_start_bikes: int = 0,
    evaluation_end_bikes: int = 5,
    evaluation_gap_hours: int = 1,
    training_rentals: int = 60,
) -> pl.DataFrame:
    timestamps: list[datetime] = []
    bikes: list[int] = []
    rentals: list[int] = []
    for day_offset in range(3):
        start = TRAINING_START + timedelta(days=day_offset)
        timestamps.extend([start, start + timedelta(hours=1)])
        bikes.extend([5, 5])
        rentals.extend([training_rentals, 0])
    timestamps.extend(
        [
            EVALUATION_START,
            EVALUATION_START + timedelta(hours=evaluation_gap_hours),
        ]
    )
    bikes.extend([evaluation_start_bikes, evaluation_end_bikes])
    rentals.extend([999, 999])
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "station_id": ["A"] * len(timestamps),
            "available_bikes": bikes,
            "rentals": rentals,
            "inventory_observed": [True] * len(timestamps),
            "actionable": [True] * len(timestamps),
        }
    )


def _trips(*, include_evaluation_rows: bool = False) -> pl.DataFrame:
    rows = []
    for index in range(20):
        rent_at = TRAINING_START + timedelta(days=index % 3, minutes=index)
        duration = 10 + index
        rows.append(
            {
                "rent_at": rent_at,
                "return_at": rent_at + timedelta(minutes=duration),
                "rent_station_id": "A",
                "return_station_id": "B" if index % 2 == 0 else "OUT",
                "duration_minutes": duration,
                "rent_in_scope": True,
            }
        )
    if include_evaluation_rows:
        for index in range(30):
            rent_at = EVALUATION_START + timedelta(seconds=index)
            rows.append(
                {
                    "rent_at": rent_at,
                    "return_at": rent_at + timedelta(minutes=999),
                    "rent_station_id": "A",
                    "return_station_id": "HOLDOUT_ONLY",
                    "duration_minutes": 999,
                    "rent_in_scope": True,
                }
            )
    return pl.DataFrame(rows)


def _request_ids(frame: pl.DataFrame) -> set[str]:
    return set(frame["request_id"].to_list())


def test_manifest_is_nested_deterministic_and_input_order_invariant() -> None:
    station_hour = _station_hour()
    trips = _trips()

    first = build_latent_demand_manifest(trips, station_hour, _config())
    repeated = build_latent_demand_manifest(trips, station_hour, _config())
    reordered = build_latent_demand_manifest(trips.reverse(), station_hour.reverse(), _config())
    other_seed = build_latent_demand_manifest(trips, station_hour, _config(seed=20251125))

    low_ids = _request_ids(first.requests_for("low"))
    base_ids = _request_ids(first.requests_for("base"))
    high_ids = _request_ids(first.requests_for("high"))
    assert low_ids
    assert low_ids < base_ids < high_ids
    assert first.audit.master_manifest_hash == repeated.audit.master_manifest_hash
    assert first.audit.master_manifest_hash == reordered.audit.master_manifest_hash
    assert first.audit.master_manifest_hash != other_seed.audit.master_manifest_hash
    assert first.audit.scenario_requests == {
        "low": len(low_ids),
        "base": len(base_ids),
        "high": len(high_ids),
    }


def test_manifest_separates_source_and_samples_joint_trip_tuple() -> None:
    manifest = build_latent_demand_manifest(_trips(), _station_hour(), _config())

    assert manifest.requests["source"].unique().to_list() == ["synthetic_latent"]
    assert manifest.requests["contract_version"].unique().to_list() == [CONTRACT_VERSION]
    assert manifest.requests["request_id"].str.starts_with("syn:").all()
    assert manifest.requests["trip_pool"].unique().to_list() == ["origin_day_hour"]
    assert manifest.requests["intensity_pool"].unique().to_list() == ["station_day_hour"]
    for row in manifest.requests.iter_rows(named=True):
        assert row["return_at"] - row["rental_at"] == timedelta(minutes=row["duration_minutes"])
        assert row["destination_station_id"] in {"B", "OUT"}


def test_evaluation_rows_do_not_tune_manifest() -> None:
    station_hour = _station_hour()
    changed_evaluation_counts = station_hour.with_columns(
        pl.when(pl.col("timestamp") >= EVALUATION_START)
        .then(pl.lit(1_000_000))
        .otherwise(pl.col("rentals"))
        .alias("rentals")
    )

    plain = build_latent_demand_manifest(_trips(), station_hour, _config())
    with_holdout_trips = build_latent_demand_manifest(
        _trips(include_evaluation_rows=True), changed_evaluation_counts, _config()
    )

    assert plain.audit.master_manifest_hash == with_holdout_trips.audit.master_manifest_hash
    assert plain.requests.equals(with_holdout_trips.requests)


@pytest.mark.parametrize(
    ("start_bikes", "end_bikes", "gap_hours"),
    [(5, 5, 1), (0, 5, 2)],
)
def test_non_censored_or_non_hourly_interval_generates_no_requests(
    start_bikes: int,
    end_bikes: int,
    gap_hours: int,
) -> None:
    manifest = build_latent_demand_manifest(
        _trips(),
        _station_hour(
            evaluation_start_bikes=start_bikes,
            evaluation_end_bikes=end_bikes,
            evaluation_gap_hours=gap_hours,
        ),
        _config(),
    )

    assert manifest.requests.is_empty()
    assert manifest.audit.evaluation_censored_intervals == 0
    assert manifest.audit.scenario_requests == {"low": 0, "base": 0, "high": 0}


def test_duplicate_station_timestamp_fails_closed() -> None:
    station_hour = _station_hour()
    duplicated = pl.concat([station_hour, station_hour.head(1)])

    with pytest.raises(LatentDemandError, match="중복"):
        build_latent_demand_manifest(_trips(), duplicated, _config())


def test_empirical_hourly_cap_is_enforced() -> None:
    station_hour = _station_hour(
        evaluation_start_bikes=0,
        evaluation_end_bikes=0,
        training_rentals=60,
    )
    manifest = build_latent_demand_manifest(_trips(), station_hour, _config())

    assert manifest.requests.height <= 60
    assert manifest.requests["hourly_cap"].unique().to_list() == [60]
    assert manifest.audit.capped_intervals == 1
    assert manifest.audit.dropped_candidates_by_cap > 0
    assert manifest.audit.generated_candidates_before_cap >= manifest.requests.height
    assert manifest.audit.dropped_candidates_by_cap == (
        manifest.audit.generated_candidates_before_cap - manifest.requests.height
    )


def test_district_backoff_is_audited_when_origin_has_no_training_rows() -> None:
    base = _station_hour()
    training = base.filter(pl.col("timestamp") < TRAINING_END)
    evaluation = base.filter(pl.col("timestamp") >= EVALUATION_START).with_columns(
        pl.lit("B").alias("station_id")
    )

    manifest = build_latent_demand_manifest(
        _trips(),
        pl.concat([training, evaluation]),
        _config(),
    )

    assert not manifest.requests.is_empty()
    assert manifest.requests["intensity_pool"].unique().to_list() == ["district_day_hour"]
    assert manifest.requests["trip_pool"].unique().to_list() == ["district_day_hour"]
    assert manifest.audit.district_fallback_rate == 1.0
