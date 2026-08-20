from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from ddareungi_rearrangement.seoul_api import LiveBikePage
from ddareungi_rearrangement.simulation import (
    FleetExecutionConfig,
    GreedyNearestPolicy,
    NoRelocationPolicy,
    SimulationConfig,
    StaticThresholdPolicy,
    build_replay_scenario,
    run_daily_policy_comparison,
    run_donor_reserve_holdout,
    run_donor_reserve_training,
    run_request_transition_trace,
    run_spatial_sensitivity,
    run_station_equity_comparison,
    select_greedy_medoids,
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


def test_greedy_nearest_donor_reserve_limits_supply_without_changing_default() -> None:
    coordinates = {"R": (37.5, 127.0), "D": (37.5, 127.01)}
    default_policy = GreedyNearestPolicy(
        coordinates=coordinates,
        max_actions_per_decision=1,
        vehicle_capacity=10,
    )
    protected_policy = GreedyNearestPolicy(
        coordinates=coordinates,
        donor_reserve_bikes=8,
        max_actions_per_decision=1,
        vehicle_capacity=10,
    )

    default_transfer = default_policy.plan({"R": 0, "D": 10}, max_bikes=40)
    protected_transfer = protected_policy.plan({"R": 0, "D": 10}, max_bikes=40)

    assert default_transfer[0].bike_count == 5
    assert protected_transfer[0].bike_count == 2


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


def test_daily_policy_comparison_resets_inventory_and_preserves_daily_contract() -> None:
    first = datetime(2025, 11, 1)
    second = datetime(2025, 11, 2)
    end = datetime(2025, 11, 3)
    trips = pl.DataFrame(
        {
            "rent_at": [
                datetime(2025, 11, 1, 0, 10),
                datetime(2025, 11, 2, 0, 10),
            ],
            "return_at": [
                datetime(2025, 11, 1, 0, 30),
                datetime(2025, 11, 2, 0, 30),
            ],
            "rent_station_id": ["A", "A"],
            "return_station_id": ["OUT", "OUT"],
        }
    )
    station_hour = pl.concat(
        [
            _station_hour(first, bikes_a=0, bikes_b=10),
            _station_hour(second, bikes_a=0, bikes_b=10),
        ]
    )

    frame, excluded = run_daily_policy_comparison(
        trips,
        station_hour,
        {"A": (37.5, 127.0), "B": (37.5, 127.001)},
        analysis_start=first,
        analysis_end=end,
    )

    assert frame.height == 6
    assert excluded == ()
    assert frame["date"].n_unique() == 2
    daily_request_counts = frame.group_by("date").agg(pl.col("observed_requests").n_unique())
    assert daily_request_counts["observed_requests"].to_list() == [1, 1]
    assert (frame["conservation_residual"] == 0).all()
    assert (
        frame.filter(pl.col("policy_name") == "greedy_default")["max_relocation_actions_in_epoch"]
        <= 2
    ).all()
    assert (
        frame.filter(pl.col("policy_name") == "greedy_service")["max_relocation_actions_in_epoch"]
        <= 3
    ).all()
    assert (
        frame.filter(pl.col("policy_name") != "no_relocation")["failures_avoided_vs_p0"] == 1
    ).all()
    assert (
        frame.filter(pl.col("policy_name") == "greedy_service")[
            "additional_failures_avoided_vs_default"
        ]
        == 0
    ).all()


def test_station_equity_comparison_preserves_requests_and_reconciles_failures() -> None:
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
        SimulationConfig(start=start, end=end, max_bikes_per_decision=40),
    )

    frame, runs = run_station_equity_comparison(
        scenario,
        {"A": (37.5, 127.0), "B": (37.5, 127.001)},
    )

    assert frame.height == 2
    assert len(runs) == 3
    assert frame["requests"].sum() == 1
    assert frame["p0_failed_rentals"].sum() == runs[0][1].metrics.failed_rentals
    assert frame["default_failed_rentals"].sum() == runs[1][1].metrics.failed_rentals
    assert frame["service_failed_rentals"].sum() == runs[2][1].metrics.failed_rentals
    assert frame["default_failures_avoided_vs_p0"].sum() == 1
    assert frame["service_failures_avoided_vs_p0"].sum() == 1
    assert all(run.metrics.conservation_residual == 0 for _, run in runs)


def test_harm_trace_links_p0_success_p2_failure_to_prior_donor_outflow() -> None:
    start = datetime(2025, 11, 24)
    end = datetime(2025, 11, 24, 1)
    trips = pl.DataFrame(
        {
            "rent_at": [datetime(2025, 11, 24, 0, minute) for minute in range(5, 11)],
            "return_at": [datetime(2025, 11, 24, 0, minute) for minute in range(30, 36)],
            "rent_station_id": ["A"] * 6,
            "return_station_id": ["OUT"] * 6,
        }
    )
    scenario = build_replay_scenario(
        trips,
        _station_hour(start, bikes_a=10, bikes_b=0),
        SimulationConfig(start=start, end=end, max_bikes_per_decision=40),
    )
    coordinates = {"A": (37.5, 127.0), "B": (37.5, 127.001)}

    plain = simulate_replay(scenario, NoRelocationPolicy())
    harm_frame, summaries, runs = run_request_transition_trace(scenario, coordinates)

    assert plain.event_trace is None
    assert harm_frame.height == 2
    assert harm_frame["has_prior_relocation_out"].all()
    assert harm_frame["within_60_minutes_of_prior_out"].all()
    assert harm_frame["prior_out_bikes"].to_list() == [5, 5]
    assert harm_frame["prior_out_inventory_after"].to_list() == [5, 5]
    assert summaries["greedy_default"]["rescued_requests"] == 0
    assert summaries["greedy_default"]["harmed_requests"] == 1
    assert summaries["greedy_default"]["net_failures_avoided"] == -1
    assert summaries["greedy_default"]["reconciliation_residual"] == 0
    assert all(run.event_trace is not None for _, run in runs)
    assert all(run.metrics.conservation_residual == 0 for _, run in runs)


def test_donor_reserve_training_marks_dominated_candidate_without_weighted_score() -> None:
    start = datetime(2025, 11, 24)
    end = datetime(2025, 11, 24, 1)
    trips = pl.DataFrame(
        {
            "rent_at": [datetime(2025, 11, 24, 0, minute) for minute in range(5, 11)],
            "return_at": [datetime(2025, 11, 24, 0, minute) for minute in range(30, 36)],
            "rent_station_id": ["A"] * 6,
            "return_station_id": ["OUT"] * 6,
        }
    )
    scenario = build_replay_scenario(
        trips,
        _station_hour(start, bikes_a=10, bikes_b=0),
        SimulationConfig(start=start, end=end, max_bikes_per_decision=40),
    )

    frame, baseline, runs = run_donor_reserve_training(
        scenario,
        {"A": (37.5, 127.0), "B": (37.5, 127.001)},
        reserves=(5, 8),
    )

    reserve_five = frame.filter(pl.col("donor_reserve_bikes") == 5).row(0, named=True)
    reserve_eight = frame.filter(pl.col("donor_reserve_bikes") == 8).row(0, named=True)
    assert baseline.metrics.failed_rentals == 0
    assert reserve_five["failed_rentals"] == 1
    assert reserve_five["harmed_requests_vs_p0"] == 1
    assert not reserve_five["is_pareto"]
    assert reserve_five["dominated_by_reserves"] == "8"
    assert reserve_eight["failed_rentals"] == 0
    assert reserve_eight["harmed_requests_vs_p0"] == 0
    assert reserve_eight["is_pareto"]
    assert all(run.metrics.conservation_residual == 0 for run in runs)


def test_donor_reserve_holdout_compares_only_frozen_policy_and_reconciles_trace() -> None:
    start = datetime(2025, 11, 24)
    end = datetime(2025, 11, 24, 1)
    trips = pl.DataFrame(
        {
            "rent_at": [datetime(2025, 11, 24, 0, minute) for minute in range(5, 11)],
            "return_at": [datetime(2025, 11, 24, 0, minute) for minute in range(30, 36)],
            "rent_station_id": ["A"] * 6,
            "return_station_id": ["OUT"] * 6,
        }
    )
    scenario = build_replay_scenario(
        trips,
        _station_hour(start, bikes_a=10, bikes_b=0),
        SimulationConfig(start=start, end=end, max_bikes_per_decision=40),
    )

    frame, runs = run_donor_reserve_holdout(
        scenario,
        {"A": (37.5, 127.0), "B": (37.5, 127.001)},
        selected_reserve=7,
    )

    rows = {row["policy_role"]: row for row in frame.to_dicts()}
    assert set(rows) == {"P0", "P2", "P3"}
    assert all(row["observed_requests"] == 6 for row in rows.values())
    assert rows["P0"]["failed_rentals"] == 0
    assert rows["P2"]["donor_reserve_bikes"] == 5
    assert rows["P2"]["failed_rentals"] == 1
    assert rows["P2"]["harmed_requests_vs_p0"] == 1
    assert rows["P3"]["donor_reserve_bikes"] == 7
    assert rows["P3"]["failed_rentals"] == 0
    assert rows["P3"]["harmed_requests_vs_p0"] == 0
    assert rows["P3"]["bikes_moved"] < rows["P2"]["bikes_moved"]
    assert all(row["transition_reconciliation_residual"] == 0 for row in rows.values())
    assert all(run.metrics.conservation_residual == 0 for run in runs)


def test_greedy_medoids_are_deterministic_and_add_the_best_next_station() -> None:
    coordinates = {
        "C": (37.5, 127.02),
        "A": (37.5, 127.0),
        "B": (37.5, 127.01),
    }

    selected = select_greedy_medoids(coordinates, 2)

    assert selected == ("B", "A")
    assert select_greedy_medoids(dict(reversed(list(coordinates.items()))), 2) == selected


def test_fleet_keeps_donor_inventory_until_pickup_and_uses_actual_surplus() -> None:
    start = datetime(2025, 11, 24)
    end = datetime(2025, 11, 24, 1)
    station_hour = pl.DataFrame(
        {
            "timestamp": [start] * 3,
            "station_id": ["A", "B", "C"],
            "station_name": ["공급", "수요", "차량"],
            "available_bikes": [10, 0, 5],
            "inventory_observed": [True] * 3,
            "actionable": [True] * 3,
        }
    )
    trips = pl.DataFrame(
        {
            "rent_at": [datetime(2025, 11, 24, 0, 1)],
            "return_at": [datetime(2025, 11, 24, 0, 30)],
            "rent_station_id": ["A"],
            "return_station_id": ["OUT"],
        }
    )
    coordinates = {
        "A": (37.5, 127.0),
        "B": (37.5, 127.001),
        "C": (37.5, 127.02),
    }
    scenario = build_replay_scenario(
        trips,
        station_hour,
        SimulationConfig(start=start, end=end, max_bikes_per_decision=40),
    )
    policy = GreedyNearestPolicy(
        coordinates=coordinates,
        donor_reserve_bikes=7,
        max_actions_per_decision=1,
        vehicle_capacity=10,
        average_speed_kmh=60,
        road_distance_factor=1,
        handling_minutes_per_bike=0,
        name="protected",
    )

    run = simulate_replay(
        scenario,
        policy,
        collect_trace=True,
        fleet_config=FleetExecutionConfig(
            coordinates=coordinates,
            fleet_size=1,
            donor_reserve_bikes=7,
            vehicle_capacity=10,
            average_speed_kmh=60,
            road_distance_factor=1,
            handling_minutes_per_bike=0,
            initial_station_ids=("C",),
        ),
    )

    assert run.event_trace is not None
    assert run.fleet_metrics is not None
    dispatch = run.event_trace.filter(pl.col("event_kind") == "vehicle_dispatch").row(0, named=True)
    rental = run.event_trace.filter(pl.col("event_kind") == "rental").row(0, named=True)
    pickup = run.event_trace.filter(pl.col("event_kind") == "relocation_out").row(0, named=True)
    assert dispatch["inventory_before"] == dispatch["inventory_after"] == 10
    assert rental["rental_outcome"] == "successful"
    assert pickup["timestamp"] > rental["timestamp"]
    assert pickup["bike_count"] == 2
    assert pickup["inventory_after"] == 7
    assert run.fleet_metrics.dispatched_planned_bikes == 3
    assert run.fleet_metrics.picked_up_bikes == 2
    assert run.fleet_metrics.pickup_shortfall_bikes == 1
    assert run.fleet_metrics.approach_distance_km > run.fleet_metrics.loaded_distance_km
    assert run.fleet_metrics.total_vehicle_distance_km == round(
        run.fleet_metrics.approach_distance_km + run.fleet_metrics.loaded_distance_km,
        3,
    )
    assert run.metrics.conservation_residual == 0


def test_busy_fleet_skips_new_intents_without_overlapping_vehicle_jobs() -> None:
    start = datetime(2025, 11, 24)
    end = datetime(2025, 11, 24, 0, 5)
    station_hour = pl.DataFrame(
        {
            "timestamp": [start] * 4,
            "station_id": ["A", "B", "C", "D"],
            "station_name": ["첫 공급", "먼 수요", "둘째 공급", "둘째 수요"],
            "available_bikes": [10, 0, 5, 0],
            "inventory_observed": [True] * 4,
            "actionable": [True] * 4,
        }
    )
    trips = pl.DataFrame(
        {
            "rent_at": [start - timedelta(hours=1)] * 4 + [datetime(2025, 11, 24, 0, 2)],
            "return_at": [
                datetime(2025, 11, 24, 0, 0, 30),
                datetime(2025, 11, 24, 0, 0, 31),
                datetime(2025, 11, 24, 0, 0, 32),
                datetime(2025, 11, 24, 0, 0, 33),
                datetime(2025, 11, 24, 0, 4),
            ],
            "rent_station_id": ["OUT"] * 4 + ["B"],
            "return_station_id": ["C"] * 4 + ["OUT"],
        }
    )
    coordinates = {
        "A": (37.5, 127.0),
        "B": (37.5, 127.1),
        "C": (37.5, 127.2),
        "D": (37.5, 127.201),
    }
    scenario = build_replay_scenario(
        trips,
        station_hour,
        SimulationConfig(
            start=start,
            end=end,
            decision_interval_minutes=1,
            max_bikes_per_decision=40,
        ),
    )
    policy = GreedyNearestPolicy(
        coordinates=coordinates,
        donor_reserve_bikes=7,
        max_actions_per_decision=1,
        vehicle_capacity=10,
        average_speed_kmh=10,
        road_distance_factor=1,
        handling_minutes_per_bike=0,
        name="protected_busy",
    )

    run = simulate_replay(
        scenario,
        policy,
        collect_trace=True,
        fleet_config=FleetExecutionConfig(
            coordinates=coordinates,
            fleet_size=1,
            donor_reserve_bikes=7,
            vehicle_capacity=10,
            average_speed_kmh=10,
            road_distance_factor=1,
            handling_minutes_per_bike=0,
            initial_station_ids=("A",),
        ),
    )

    assert run.fleet_metrics is not None
    assert run.fleet_metrics.jobs_dispatched == 1
    assert run.fleet_metrics.jobs_cancelled_no_vehicle >= 1
    assert run.fleet_metrics.vehicles_busy_at_end == 1
    assert run.fleet_metrics.vehicle_summaries[0]["jobs_dispatched"] == 1
    assert run.fleet_metrics.vehicle_summaries[0]["jobs_completed"] == 0
    assert run.metrics.relocation_bikes_in_transit_at_end == 3
    assert run.metrics.max_relocation_actions_in_epoch == 1
    assert run.metrics.conservation_residual == 0


def test_fleet_next_approach_starts_from_previous_delivery_station() -> None:
    start = datetime(2025, 11, 24)
    end = datetime(2025, 11, 24, 2)
    station_hour = pl.DataFrame(
        {
            "timestamp": [start] * 4,
            "station_id": ["A", "B", "C", "D"],
            "station_name": ["첫 공급", "첫 수요", "둘째 공급", "둘째 수요"],
            "available_bikes": [10, 0, 5, 0],
            "inventory_observed": [True] * 4,
            "actionable": [True] * 4,
        }
    )
    trips = pl.DataFrame(
        {
            "rent_at": [start - timedelta(hours=1)] * 4 + [datetime(2025, 11, 24, 1, 5)],
            "return_at": [
                datetime(2025, 11, 24, 0, 30),
                datetime(2025, 11, 24, 0, 31),
                datetime(2025, 11, 24, 0, 32),
                datetime(2025, 11, 24, 0, 33),
                datetime(2025, 11, 24, 1, 30),
            ],
            "rent_station_id": ["OUT"] * 4 + ["D"],
            "return_station_id": ["C"] * 4 + ["OUT"],
        }
    )
    coordinates = {
        "A": (37.5, 127.0),
        "B": (37.5, 127.001),
        "C": (37.5, 127.01),
        "D": (37.5, 127.011),
    }
    scenario = build_replay_scenario(
        trips,
        station_hour,
        SimulationConfig(
            start=start,
            end=end,
            decision_interval_minutes=60,
            max_bikes_per_decision=40,
        ),
    )
    policy = GreedyNearestPolicy(
        coordinates=coordinates,
        donor_reserve_bikes=7,
        max_actions_per_decision=1,
        vehicle_capacity=10,
        average_speed_kmh=60,
        road_distance_factor=1,
        handling_minutes_per_bike=0,
        name="protected_continuous",
    )

    run = simulate_replay(
        scenario,
        policy,
        collect_trace=True,
        fleet_config=FleetExecutionConfig(
            coordinates=coordinates,
            fleet_size=1,
            donor_reserve_bikes=7,
            vehicle_capacity=10,
            average_speed_kmh=60,
            road_distance_factor=1,
            handling_minutes_per_bike=0,
            initial_station_ids=("A",),
        ),
    )

    assert run.event_trace is not None
    assert run.fleet_metrics is not None
    dispatches = run.event_trace.filter(pl.col("event_kind") == "vehicle_dispatch").sort(
        "timestamp"
    )
    assert dispatches.height == 2
    assert dispatches["vehicle_id"].to_list() == ["V01", "V01"]
    assert dispatches["station_id"].to_list() == ["A", "C"]
    assert dispatches["distance_km"].to_list()[0] == 0
    assert dispatches["distance_km"].to_list()[1] > 0
    vehicle = run.fleet_metrics.vehicle_summaries[0]
    assert vehicle["jobs_completed"] == 2
    assert vehicle["final_station_id"] == "D"
    assert vehicle["status_at_end"] == "idle"
    assert run.metrics.failed_rentals == 0
    assert run.metrics.conservation_residual == 0
