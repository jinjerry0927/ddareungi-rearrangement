from datetime import datetime
from pathlib import Path

import polars as pl

from ddareungi_rearrangement.seoul_api import LiveBikePage
from ddareungi_rearrangement.simulation import (
    GreedyNearestPolicy,
    NoRelocationPolicy,
    SimulationConfig,
    StaticThresholdPolicy,
    build_replay_scenario,
    run_spatial_sensitivity,
    simulate_replay,
    snapshot_actionable_coordinates,
)


def _station_hour(start: datetime, bikes_a: int, bikes_b: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [start, start],
            "station_id": ["A", "B"],
            "station_name": ["에이", "비"],
            "available_bikes": [bikes_a, bikes_b],
            "inventory_observed": [True, True],
            "actionable": [True, True],
        }
    )


def test_failed_rental_suppresses_return_and_unconditional_inbound_still_arrives() -> None:
    start = datetime(2025, 11, 24)
    end = datetime(2025, 11, 24, 1)
    trips = pl.DataFrame(
        {
            "rent_at": [
                datetime(2025, 11, 24, 0, 10),
                datetime(2025, 11, 23, 23, 50),
                datetime(2025, 11, 24, 0, 30),
            ],
            "return_at": [
                datetime(2025, 11, 24, 0, 20),
                datetime(2025, 11, 24, 0, 15),
                datetime(2025, 11, 24, 0, 40),
            ],
            "rent_station_id": ["A", "OUT", "A"],
            "return_station_id": ["B", "A", "B"],
        }
    )
    scenario = build_replay_scenario(
        trips,
        _station_hour(start, bikes_a=0, bikes_b=2),
        SimulationConfig(start=start, end=end),
    )

    run = simulate_replay(scenario, NoRelocationPolicy())

    assert run.metrics.observed_requests == 2
    assert run.metrics.successful_rentals == 1
    assert run.metrics.failed_rentals == 1
    assert run.metrics.unconditional_inbound_returns == 1
    assert run.metrics.successful_internal_returns == 1
    assert run.metrics.final_station_bikes == 3
    assert run.metrics.empty_station_minutes == 45
    assert run.metrics.conservation_residual == 0


def test_static_threshold_policy_respects_epoch_budget() -> None:
    policy = StaticThresholdPolicy(
        lower_threshold=2,
        target_bikes=5,
        upper_threshold=8,
    )

    transfers = policy.plan({"A": 0, "B": 10, "C": 12}, max_bikes=3)

    assert sum(transfer.bike_count for transfer in transfers) == 3
    assert transfers[0].from_station_id == "C"
    assert transfers[0].to_station_id == "A"


def test_threshold_relocation_preserves_bikes_and_can_prevent_failure() -> None:
    start = datetime(2025, 11, 24)
    end = datetime(2025, 11, 24, 1)
    trips = pl.DataFrame(
        {
            "rent_at": [datetime(2025, 11, 24, 0, 10)],
            "return_at": [datetime(2025, 11, 24, 0, 30)],
            "rent_station_id": ["A"],
            "return_station_id": ["OUT"],
        }
    )
    scenario = build_replay_scenario(
        trips,
        _station_hour(start, bikes_a=0, bikes_b=10),
        SimulationConfig(start=start, end=end, max_bikes_per_decision=3),
    )

    baseline = simulate_replay(scenario, NoRelocationPolicy())
    threshold = simulate_replay(
        scenario,
        StaticThresholdPolicy(
            lower_threshold=2,
            target_bikes=5,
            upper_threshold=8,
        ),
    )

    assert baseline.metrics.failed_rentals == 1
    assert threshold.metrics.failed_rentals == 0
    assert threshold.metrics.bikes_moved == 3
    assert threshold.metrics.max_bikes_moved_in_epoch == 3
    assert threshold.metrics.conservation_residual == 0
    assert threshold.metrics.observed_requests == baseline.metrics.observed_requests


def test_coordinate_snapshot_maps_display_ids_and_reports_missing(tmp_path: Path) -> None:
    station_hour_path = tmp_path / "station_hour.parquet"
    output_path = tmp_path / "coordinates.csv"
    pl.DataFrame(
        {
            "station_id": ["100", "200"],
            "station_name": ["백", "이백"],
            "actionable": [True, True],
        }
    ).write_parquet(station_hour_path)
    page = LiveBikePage(
        start=1,
        end=1,
        response_service_name="rentBikeStatus",
        result_code="INFO-000",
        reported_count=1,
        rows=(
            {
                "stationId": "ST-A",
                "stationName": "100. 백 최신명",
                "stationLatitude": "37.5",
                "stationLongitude": "127.0",
            },
        ),
    )

    snapshot = snapshot_actionable_coordinates(
        [page],
        station_hour_path=station_hour_path,
        output_path=output_path,
        captured_at_utc=datetime(2026, 8, 20),
    )

    coordinates = pl.read_csv(output_path)
    assert snapshot.matched_stations == 1
    assert snapshot.missing_station_ids == ("200",)
    assert coordinates["station_id"].to_list() == [100]
    assert coordinates["latitude"].to_list() == [37.5]


def test_greedy_nearest_chooses_closest_donor_and_limits_actions() -> None:
    policy = GreedyNearestPolicy(
        coordinates={
            "R": (37.5, 127.0),
            "NEAR": (37.5, 127.01),
            "FAR": (37.5, 127.1),
        },
        max_actions_per_decision=1,
        vehicle_capacity=3,
    )

    transfers = policy.plan({"R": 0, "NEAR": 10, "FAR": 20}, max_bikes=40)

    assert len(transfers) == 1
    assert transfers[0].from_station_id == "NEAR"
    assert transfers[0].to_station_id == "R"
    assert transfers[0].bike_count == 3
    assert transfers[0].distance_km > 0
    assert transfers[0].travel_minutes > 0


def test_delayed_relocation_arrival_changes_request_outcome_and_preserves_bikes() -> None:
    start = datetime(2025, 11, 24)
    end = datetime(2025, 11, 24, 1)
    trips = pl.DataFrame(
        {
            "rent_at": [
                datetime(2025, 11, 24, 0, 0, 30),
                datetime(2025, 11, 24, 0, 2),
            ],
            "return_at": [
                datetime(2025, 11, 24, 0, 10),
                datetime(2025, 11, 24, 0, 12),
            ],
            "rent_station_id": ["A", "A"],
            "return_station_id": ["OUT", "OUT"],
        }
    )
    scenario = build_replay_scenario(
        trips,
        _station_hour(start, bikes_a=0, bikes_b=10),
        SimulationConfig(start=start, end=end, max_bikes_per_decision=3),
    )
    policy = GreedyNearestPolicy(
        coordinates={"A": (37.5, 127.0), "B": (37.5, 127.01)},
        max_actions_per_decision=1,
        vehicle_capacity=3,
        average_speed_kmh=60,
        road_distance_factor=1,
        handling_minutes_per_bike=0,
    )

    run = simulate_replay(scenario, policy)

    assert run.metrics.observed_requests == 2
    assert run.metrics.failed_rentals == 1
    assert run.metrics.successful_rentals == 1
    assert run.metrics.bikes_moved == 3
    assert run.metrics.max_relocation_actions_in_epoch == 1
    assert run.metrics.relocation_distance_km > 0
    assert run.metrics.relocation_bikes_in_transit_at_end == 0
    assert run.metrics.conservation_residual == 0


def test_spatial_sensitivity_runs_full_grid_with_shared_requests_and_constraints() -> None:
    start = datetime(2025, 11, 24)
    end = datetime(2025, 11, 24, 1)
    trips = pl.DataFrame(
        {
            "rent_at": [
                datetime(2025, 11, 24, 0, 2),
                datetime(2025, 11, 24, 0, 3),
            ],
            "return_at": [
                datetime(2025, 11, 24, 0, 20),
                datetime(2025, 11, 24, 0, 21),
            ],
            "rent_station_id": ["A", "A"],
            "return_station_id": ["OUT", "OUT"],
        }
    )
    scenario = build_replay_scenario(
        trips,
        _station_hour(start, bikes_a=0, bikes_b=10),
        SimulationConfig(start=start, end=end, max_bikes_per_decision=4),
    )

    frame = run_spatial_sensitivity(
        scenario,
        {"A": (37.5, 127.0), "B": (37.5, 127.001)},
        action_counts=(1, 2),
        speeds_kmh=(15.0,),
        vehicle_capacities=(2,),
    )

    assert frame.height == 2
    assert frame["observed_requests"].unique().to_list() == [2]
    assert frame["conservation_residual"].unique().to_list() == [0]
    assert frame["relocation_bikes_in_transit_at_end"].unique().to_list() == [0]
    assert (frame["max_relocation_actions_in_epoch"] <= frame["max_actions_per_decision"]).all()
    assert (frame["relocation_distance_km"] > 0).all()
