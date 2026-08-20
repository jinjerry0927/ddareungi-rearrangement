from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from ddareungi_rearrangement.doctor import format_report, inspect_environment
from ddareungi_rearrangement.historical_audit import (
    HistoricalDataError,
    audit_inventory_zip,
    audit_rental_pages,
    audit_station_workbook,
    build_historical_audit,
    write_historical_reports,
)
from ddareungi_rearrangement.live_audit import audit_live_bike_pages, write_audit_reports
from ddareungi_rearrangement.pilot_analysis import (
    PilotAnalysisError,
    build_pilot_baseline,
    extract_borough_inventory,
    extract_borough_trips,
    write_pilot_reports,
)
from ddareungi_rearrangement.seoul_api import SeoulOpenDataClient, SeoulOpenDataError
from ddareungi_rearrangement.simulation import (
    SimulationError,
    build_donor_reserve_holdout,
    build_donor_reserve_training,
    build_fleet_sensitivity,
    build_harm_trace,
    build_policy_comparison,
    build_spatial_policy_comparison,
    build_spatial_sensitivity,
    build_station_equity,
    build_temporal_robustness,
    snapshot_actionable_coordinates,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ddareungi",
        description="따릉이 재배치 분석 프로젝트 도구",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "doctor",
        help="로컬 프로젝트 환경을 점검합니다.",
    )
    audit_parser = subparsers.add_parser(
        "audit-live-api",
        help="실시간 따릉이 API 연결과 스키마를 감사합니다.",
    )
    audit_parser.add_argument("--page-size", type=int, default=1_000)
    audit_parser.add_argument("--max-pages", type=int, default=10)
    audit_parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/live_api_schema.json"),
    )
    audit_parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/live_api_schema.md"),
    )
    historical_parser = subparsers.add_parser(
        "audit-historical-data",
        help="과거 대여이력·재고·대여소 메타데이터를 함께 감사합니다.",
    )
    historical_parser.add_argument(
        "--station-file",
        type=Path,
        default=Path("data/raw/stations_2025_12.xlsx"),
    )
    historical_parser.add_argument(
        "--inventory-file",
        type=Path,
        default=Path("data/raw/inventory_2025_q4.zip"),
    )
    historical_parser.add_argument("--target-month", default="2025-11")
    historical_parser.add_argument("--rental-date", default="2025-11-05")
    historical_parser.add_argument("--rental-hour", type=int)
    historical_parser.add_argument("--page-size", type=int, default=1_000)
    historical_parser.add_argument("--max-pages", type=int, default=100)
    historical_parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/historical_data_audit.json"),
    )
    historical_parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/historical_data_audit.md"),
    )
    pilot_parser = subparsers.add_parser(
        "build-pilot",
        help="한 달·한 자치구의 대여 흐름과 재고 부족 기준선을 생성합니다.",
    )
    pilot_parser.add_argument("--month", default="2025-11")
    pilot_parser.add_argument("--borough", default="강남구")
    pilot_parser.add_argument(
        "--station-file",
        type=Path,
        default=Path("data/raw/stations_2025_12.xlsx"),
    )
    pilot_parser.add_argument(
        "--rental-file",
        type=Path,
        default=Path("data/raw/rental_history_2025.zip"),
    )
    pilot_parser.add_argument(
        "--inventory-file",
        type=Path,
        default=Path("data/raw/inventory_2025_q4.zip"),
    )
    pilot_parser.add_argument(
        "--trips-output",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_trips.parquet"),
    )
    pilot_parser.add_argument(
        "--inventory-output",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_inventory.parquet"),
    )
    pilot_parser.add_argument(
        "--station-hour-output",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_station_hour.parquet"),
    )
    pilot_parser.add_argument(
        "--hourly-output",
        type=Path,
        default=Path("reports/data/gangnam_2025_11_hourly_profile.csv"),
    )
    pilot_parser.add_argument(
        "--station-ranking-output",
        type=Path,
        default=Path("reports/data/gangnam_2025_11_station_ranking.csv"),
    )
    pilot_parser.add_argument(
        "--hourly-figure",
        type=Path,
        default=Path("reports/figures/gangnam_2025_11_hourly_profile.png"),
    )
    pilot_parser.add_argument(
        "--station-figure",
        type=Path,
        default=Path("reports/figures/gangnam_2025_11_station_ranking.png"),
    )
    pilot_parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/gangnam_2025_11_baseline.json"),
    )
    pilot_parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/gangnam_2025_11_baseline.md"),
    )
    simulation_parser = subparsers.add_parser(
        "run-simulation",
        help="강남구 관측수요 재생으로 무재배치와 고정 임계값 정책을 비교합니다.",
    )
    simulation_parser.add_argument(
        "--trips-file",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_trips.parquet"),
    )
    simulation_parser.add_argument(
        "--station-hour-file",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_station_hour.parquet"),
    )
    simulation_parser.add_argument("--training-start", default="2025-11-03")
    simulation_parser.add_argument("--training-end", default="2025-11-22")
    simulation_parser.add_argument("--evaluation-start", default="2025-11-24")
    simulation_parser.add_argument("--evaluation-end", default="2025-11-29")
    simulation_parser.add_argument("--decision-interval-minutes", type=int, default=60)
    simulation_parser.add_argument("--max-bikes-per-decision", type=int, default=40)
    simulation_parser.add_argument(
        "--training-output",
        type=Path,
        default=Path("reports/data/gangnam_2025_11_simulation_training.csv"),
    )
    simulation_parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("reports/data/gangnam_2025_11_policy_comparison.csv"),
    )
    simulation_parser.add_argument(
        "--station-output",
        type=Path,
        default=Path("reports/data/gangnam_2025_11_simulation_stations.csv"),
    )
    simulation_parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("reports/figures/gangnam_2025_11_policy_comparison.png"),
    )
    simulation_parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/gangnam_2025_11_simulation.json"),
    )
    simulation_parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/gangnam_2025_11_simulation.md"),
    )
    coordinate_parser = subparsers.add_parser(
        "snapshot-coordinates",
        help="실시간 API에서 분석 대상 대여소 좌표 스냅샷을 생성합니다.",
    )
    coordinate_parser.add_argument("--page-size", type=int, default=1_000)
    coordinate_parser.add_argument("--max-pages", type=int, default=10)
    coordinate_parser.add_argument(
        "--station-hour-file",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_station_hour.parquet"),
    )
    coordinate_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sample/gangnam_station_coordinates_2026_08_20.csv"),
    )
    spatial_parser = subparsers.add_parser(
        "run-spatial-simulation",
        help="동일 대여소에서 P0/P1과 거리·시간 기반 P2를 비교합니다.",
    )
    spatial_parser.add_argument(
        "--trips-file",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_trips.parquet"),
    )
    spatial_parser.add_argument(
        "--station-hour-file",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_station_hour.parquet"),
    )
    spatial_parser.add_argument(
        "--coordinate-file",
        type=Path,
        default=Path("data/sample/gangnam_station_coordinates_2026_08_20.csv"),
    )
    spatial_parser.add_argument("--evaluation-start", default="2025-11-24")
    spatial_parser.add_argument("--evaluation-end", default="2025-11-29")
    spatial_parser.add_argument("--decision-interval-minutes", type=int, default=60)
    spatial_parser.add_argument("--max-bikes-per-decision", type=int, default=40)
    spatial_parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("reports/data/gangnam_2025_11_spatial_policy_comparison.csv"),
    )
    spatial_parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("reports/figures/gangnam_2025_11_spatial_policy_comparison.png"),
    )
    spatial_parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/gangnam_2025_11_spatial_simulation.json"),
    )
    spatial_parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/gangnam_2025_11_spatial_simulation.md"),
    )
    sensitivity_parser = subparsers.add_parser(
        "run-sensitivity",
        help="P2 직접 운송 횟수·속도·적재량의 전 요인 조합을 비교합니다.",
    )
    sensitivity_parser.add_argument(
        "--trips-file",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_trips.parquet"),
    )
    sensitivity_parser.add_argument(
        "--station-hour-file",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_station_hour.parquet"),
    )
    sensitivity_parser.add_argument(
        "--coordinate-file",
        type=Path,
        default=Path("data/sample/gangnam_station_coordinates_2026_08_20.csv"),
    )
    sensitivity_parser.add_argument("--evaluation-start", default="2025-11-24")
    sensitivity_parser.add_argument("--evaluation-end", default="2025-11-29")
    sensitivity_parser.add_argument("--decision-interval-minutes", type=int, default=60)
    sensitivity_parser.add_argument(
        "--action-counts",
        type=int,
        nargs="+",
        default=[1, 2, 3],
    )
    sensitivity_parser.add_argument(
        "--speeds-kmh",
        type=float,
        nargs="+",
        default=[10.0, 15.0, 20.0],
    )
    sensitivity_parser.add_argument(
        "--vehicle-capacities",
        type=int,
        nargs="+",
        default=[10, 20],
    )
    sensitivity_parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("reports/data/gangnam_2025_11_p2_sensitivity.csv"),
    )
    sensitivity_parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("reports/figures/gangnam_2025_11_p2_sensitivity.png"),
    )
    sensitivity_parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/gangnam_2025_11_p2_sensitivity.json"),
    )
    sensitivity_parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/gangnam_2025_11_p2_sensitivity.md"),
    )
    temporal_parser = subparsers.add_parser(
        "run-temporal-robustness",
        help="P0와 대표 P2 정책을 날짜별로 재생해 시간적 강건성을 비교합니다.",
    )
    temporal_parser.add_argument(
        "--trips-file",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_trips.parquet"),
    )
    temporal_parser.add_argument(
        "--station-hour-file",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_station_hour.parquet"),
    )
    temporal_parser.add_argument(
        "--coordinate-file",
        type=Path,
        default=Path("data/sample/gangnam_station_coordinates_2026_08_20.csv"),
    )
    temporal_parser.add_argument("--analysis-start", default="2025-11-01")
    temporal_parser.add_argument("--analysis-end", default="2025-12-01")
    temporal_parser.add_argument(
        "--daily-output",
        type=Path,
        default=Path("reports/data/gangnam_2025_11_daily_robustness.csv"),
    )
    temporal_parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("reports/figures/gangnam_2025_11_daily_robustness.png"),
    )
    temporal_parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/gangnam_2025_11_daily_robustness.json"),
    )
    temporal_parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/gangnam_2025_11_daily_robustness.md"),
    )
    equity_parser = subparsers.add_parser(
        "run-station-equity",
        help="연속 홀드아웃에서 P2의 대여소별 개선·악화 집중도를 분석합니다.",
    )
    equity_parser.add_argument(
        "--trips-file",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_trips.parquet"),
    )
    equity_parser.add_argument(
        "--station-hour-file",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_station_hour.parquet"),
    )
    equity_parser.add_argument(
        "--coordinate-file",
        type=Path,
        default=Path("data/sample/gangnam_station_coordinates_2026_08_20.csv"),
    )
    equity_parser.add_argument("--evaluation-start", default="2025-11-24")
    equity_parser.add_argument("--evaluation-end", default="2025-11-29")
    equity_parser.add_argument(
        "--station-output",
        type=Path,
        default=Path("reports/data/gangnam_2025_11_station_equity.csv"),
    )
    equity_parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("reports/figures/gangnam_2025_11_station_equity.png"),
    )
    equity_parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/gangnam_2025_11_station_equity.json"),
    )
    equity_parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/gangnam_2025_11_station_equity.md"),
    )
    harm_parser = subparsers.add_parser(
        "run-harm-trace",
        help="P0 성공에서 P2 실패로 바뀐 요청과 선행 재배치 유출을 추적합니다.",
    )
    harm_parser.add_argument(
        "--trips-file",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_trips.parquet"),
    )
    harm_parser.add_argument(
        "--station-hour-file",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_station_hour.parquet"),
    )
    harm_parser.add_argument(
        "--coordinate-file",
        type=Path,
        default=Path("data/sample/gangnam_station_coordinates_2026_08_20.csv"),
    )
    harm_parser.add_argument("--evaluation-start", default="2025-11-24")
    harm_parser.add_argument("--evaluation-end", default="2025-11-29")
    harm_parser.add_argument(
        "--harm-output",
        type=Path,
        default=Path("reports/data/gangnam_2025_11_harm_trace.csv"),
    )
    harm_parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("reports/figures/gangnam_2025_11_harm_trace.png"),
    )
    harm_parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/gangnam_2025_11_harm_trace.json"),
    )
    harm_parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/gangnam_2025_11_harm_trace.md"),
    )
    reserve_parser = subparsers.add_parser(
        "run-donor-reserve-training",
        help="학습기간에서 공급지 보유 하한 후보의 Pareto 관계를 비교합니다.",
    )
    reserve_parser.add_argument(
        "--trips-file",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_trips.parquet"),
    )
    reserve_parser.add_argument(
        "--station-hour-file",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_station_hour.parquet"),
    )
    reserve_parser.add_argument(
        "--coordinate-file",
        type=Path,
        default=Path("data/sample/gangnam_station_coordinates_2026_08_20.csv"),
    )
    reserve_parser.add_argument("--training-start", default="2025-11-03")
    reserve_parser.add_argument("--training-end", default="2025-11-22")
    reserve_parser.add_argument(
        "--reserves",
        type=int,
        nargs="+",
        default=[5, 6, 7, 8],
    )
    reserve_parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("reports/data/gangnam_2025_11_donor_reserve_training.csv"),
    )
    reserve_parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("reports/figures/gangnam_2025_11_donor_reserve_training.png"),
    )
    reserve_parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/gangnam_2025_11_donor_reserve_training.json"),
    )
    reserve_parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/gangnam_2025_11_donor_reserve_training.md"),
    )
    holdout_parser = subparsers.add_parser(
        "run-donor-reserve-holdout",
        help="동결한 공급지 보유 하한을 단일 홀드아웃에서 검증합니다.",
    )
    holdout_parser.add_argument(
        "--trips-file",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_trips.parquet"),
    )
    holdout_parser.add_argument(
        "--station-hour-file",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_station_hour.parquet"),
    )
    holdout_parser.add_argument(
        "--coordinate-file",
        type=Path,
        default=Path("data/sample/gangnam_station_coordinates_2026_08_20.csv"),
    )
    holdout_parser.add_argument("--evaluation-start", default="2025-11-24")
    holdout_parser.add_argument("--evaluation-end", default="2025-11-29")
    holdout_parser.add_argument("--selected-reserve", type=int, default=7)
    holdout_parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("reports/data/gangnam_2025_11_donor_reserve_holdout.csv"),
    )
    holdout_parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("reports/figures/gangnam_2025_11_donor_reserve_holdout.png"),
    )
    holdout_parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/gangnam_2025_11_donor_reserve_holdout.json"),
    )
    holdout_parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/gangnam_2025_11_donor_reserve_holdout.md"),
    )
    fleet_parser = subparsers.add_parser(
        "run-fleet-sensitivity",
        help="P2-R의 즉시출발과 영속 fleet 1~3대 실행 범위를 비교합니다.",
    )
    fleet_parser.add_argument(
        "--trips-file",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_trips.parquet"),
    )
    fleet_parser.add_argument(
        "--station-hour-file",
        type=Path,
        default=Path("data/processed/gangnam_2025_11_station_hour.parquet"),
    )
    fleet_parser.add_argument(
        "--coordinate-file",
        type=Path,
        default=Path("data/sample/gangnam_station_coordinates_2026_08_20.csv"),
    )
    fleet_parser.add_argument("--evaluation-start", default="2025-11-24")
    fleet_parser.add_argument("--evaluation-end", default="2025-11-29")
    fleet_parser.add_argument(
        "--fleet-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 3],
    )
    fleet_parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("reports/data/gangnam_2025_11_fleet_sensitivity.csv"),
    )
    fleet_parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("reports/figures/gangnam_2025_11_fleet_sensitivity.png"),
    )
    fleet_parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/gangnam_2025_11_fleet_sensitivity.json"),
    )
    fleet_parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("reports/gangnam_2025_11_fleet_sensitivity.md"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        report = inspect_environment()
        print(format_report(report))
        return 0 if report.ready else 1
    if args.command == "audit-live-api":
        try:
            with SeoulOpenDataClient.from_env() as client:
                pages = list(
                    client.iter_all_live_bike_pages(
                        page_size=args.page_size,
                        max_pages=args.max_pages,
                    )
                )
        except (SeoulOpenDataError, ValueError) as exc:
            print(f"Live API audit failed: {exc}")
            return 1

        audit = audit_live_bike_pages(pages)
        write_audit_reports(
            audit,
            json_path=args.json_output,
            markdown_path=args.markdown_output,
        )
        print(f"Live API audit: {audit.status}")
        print(f"Rows: {audit.total_rows:,}")
        print(f"Unique station IDs: {audit.unique_station_ids:,}")
        print(f"Report: {args.markdown_output}")
        return 0 if audit.status == "PASS" else 1
    if args.command == "audit-historical-data":
        try:
            station_dataset = audit_station_workbook(args.station_file)
            inventory = audit_inventory_zip(
                args.inventory_file,
                station_dataset.stations,
                target_month=args.target_month,
            )
            rental_hour = args.rental_hour
            if rental_hour is None:
                rental_hour = inventory.candidate_peak_hours[0]
            with SeoulOpenDataClient.from_env() as client:
                pages = list(
                    client.iter_all_rental_history_pages(
                        rent_date=args.rental_date,
                        rent_hour=rental_hour,
                        page_size=args.page_size,
                        max_pages=args.max_pages,
                    )
                )
            rental = audit_rental_pages(
                pages,
                station_dataset.stations,
                candidate_borough=inventory.candidate_borough,
            )
            audit = build_historical_audit(station_dataset.audit, inventory, rental)
            write_historical_reports(
                audit,
                json_path=args.json_output,
                markdown_path=args.markdown_output,
            )
        except (HistoricalDataError, SeoulOpenDataError, ValueError, OSError) as exc:
            print(f"Historical data audit failed: {exc}")
            return 1

        print(f"Historical data audit: {audit.status}")
        print(f"Analysis ready: {audit.analysis_ready}")
        print(f"Simulator ready: {audit.simulator_ready}")
        print(f"Selected period: {audit.selected_period}")
        print(f"Candidate borough: {audit.candidate_borough}")
        print(f"Report: {args.markdown_output}")
        return 0 if audit.analysis_ready else 1
    if args.command == "build-pilot":
        try:
            station_dataset = audit_station_workbook(args.station_file)
            trip_audit = extract_borough_trips(
                args.rental_file,
                station_dataset.stations,
                month=args.month,
                borough=args.borough,
                output_path=args.trips_output,
            )
            inventory_audit = extract_borough_inventory(
                args.inventory_file,
                station_dataset.stations,
                month=args.month,
                borough=args.borough,
                output_path=args.inventory_output,
            )
            baseline = build_pilot_baseline(
                station_dataset.stations,
                borough=args.borough,
                month=args.month,
                trip_audit=trip_audit,
                inventory_audit=inventory_audit,
                trips_path=args.trips_output,
                inventory_path=args.inventory_output,
                station_hour_path=args.station_hour_output,
                hourly_csv_path=args.hourly_output,
                station_csv_path=args.station_ranking_output,
                hourly_figure_path=args.hourly_figure,
                station_figure_path=args.station_figure,
            )
            write_pilot_reports(
                baseline,
                json_path=args.json_output,
                markdown_path=args.markdown_output,
            )
        except (HistoricalDataError, PilotAnalysisError, ValueError, OSError) as exc:
            print(f"Pilot build failed: {exc}")
            return 1

        print(f"Pilot baseline: {baseline.month} {baseline.borough}")
        print(f"Scoped trips: {baseline.scoped_trips:,}")
        print(f"Empty station-hour rate: {baseline.empty_station_hour_rate:.2%}")
        print(f"Report: {args.markdown_output}")
        return 0
    if args.command == "snapshot-coordinates":
        try:
            with SeoulOpenDataClient.from_env() as client:
                pages = list(
                    client.iter_all_live_bike_pages(
                        page_size=args.page_size,
                        max_pages=args.max_pages,
                    )
                )
            snapshot = snapshot_actionable_coordinates(
                pages,
                station_hour_path=args.station_hour_file,
                output_path=args.output,
            )
        except (SeoulOpenDataError, SimulationError, ValueError, OSError) as exc:
            print(f"Coordinate snapshot failed: {exc}")
            return 1

        print(
            f"Coordinate coverage: {snapshot.matched_stations}/{snapshot.requested_stations} "
            f"({snapshot.coverage_rate:.2%})"
        )
        print(f"Missing station IDs: {', '.join(snapshot.missing_station_ids) or 'none'}")
        print(f"Output: {snapshot.output_file}")
        return 0
    if args.command == "run-spatial-simulation":
        try:
            experiment = build_spatial_policy_comparison(
                trips_path=args.trips_file,
                station_hour_path=args.station_hour_file,
                coordinate_path=args.coordinate_file,
                evaluation_start=datetime.fromisoformat(args.evaluation_start),
                evaluation_end=datetime.fromisoformat(args.evaluation_end),
                decision_interval_minutes=args.decision_interval_minutes,
                max_bikes_per_decision=args.max_bikes_per_decision,
                comparison_csv_path=args.comparison_output,
                figure_path=args.figure_output,
                json_path=args.json_output,
                markdown_path=args.markdown_output,
            )
        except (SimulationError, ValueError, OSError) as exc:
            print(f"Spatial simulation failed: {exc}")
            return 1

        p2 = experiment.comparisons["greedy_nearest"]
        print(f"Spatial comparison stations: {experiment.stations}")
        print(f"P2 failures avoided vs P0: {p2['failures_avoided_vs_p0']:,.0f}")
        print(f"P2 additional failures vs P1: {p2['additional_failures_vs_p1']:,.0f}")
        print(f"Report: {args.markdown_output}")
        return 0
    if args.command == "run-sensitivity":
        try:
            experiment = build_spatial_sensitivity(
                trips_path=args.trips_file,
                station_hour_path=args.station_hour_file,
                coordinate_path=args.coordinate_file,
                evaluation_start=datetime.fromisoformat(args.evaluation_start),
                evaluation_end=datetime.fromisoformat(args.evaluation_end),
                decision_interval_minutes=args.decision_interval_minutes,
                action_counts=tuple(args.action_counts),
                speeds_kmh=tuple(args.speeds_kmh),
                vehicle_capacities=tuple(args.vehicle_capacities),
                comparison_csv_path=args.comparison_output,
                figure_path=args.figure_output,
                json_path=args.json_output,
                markdown_path=args.markdown_output,
            )
        except (SimulationError, ValueError, OSError) as exc:
            print(f"Sensitivity simulation failed: {exc}")
            return 1

        best = experiment.service_best
        efficient = experiment.distance_efficiency_best
        print(f"Sensitivity combinations: {len(experiment.runs)}")
        print(
            "Service best: "
            f"{best['max_actions_per_decision']} actions, "
            f"capacity {best['equivalent_vehicle_capacities']}, "
            f"{best['average_speed_kmh']:g} km/h, "
            f"{best['failed_rentals']} failures"
        )
        print(
            "Distance efficiency best: "
            f"{efficient['failures_avoided_per_100km']:,.1f} avoided failures/100km"
        )
        print(f"Report: {args.markdown_output}")
        return 0
    if args.command == "run-temporal-robustness":
        try:
            experiment = build_temporal_robustness(
                trips_path=args.trips_file,
                station_hour_path=args.station_hour_file,
                coordinate_path=args.coordinate_file,
                analysis_start=datetime.fromisoformat(args.analysis_start),
                analysis_end=datetime.fromisoformat(args.analysis_end),
                daily_csv_path=args.daily_output,
                figure_path=args.figure_output,
                json_path=args.json_output,
                markdown_path=args.markdown_output,
            )
        except (SimulationError, ValueError, OSError) as exc:
            print(f"Temporal robustness simulation failed: {exc}")
            return 1

        default = experiment.effect_consistency["greedy_default"]
        service = experiment.effect_consistency["greedy_service"]
        incremental = experiment.effect_consistency["greedy_service_vs_default"]
        print(f"Valid daily simulations: {experiment.valid_days}")
        print(f"Excluded days: {len(experiment.excluded_days)}")
        print(
            "P2 default better/tied/worse days: "
            f"{default['days_better_than_p0']}/"
            f"{default['days_tied_with_p0']}/"
            f"{default['days_worse_than_p0']}"
        )
        print(
            "P2 service better/tied/worse days: "
            f"{service['days_better_than_p0']}/"
            f"{service['days_tied_with_p0']}/"
            f"{service['days_worse_than_p0']}"
        )
        print(
            "P2 service vs default better/tied/worse days: "
            f"{incremental['days_better_than_default']}/"
            f"{incremental['days_tied_with_default']}/"
            f"{incremental['days_worse_than_default']}"
        )
        print(f"Report: {args.markdown_output}")
        return 0
    if args.command == "run-station-equity":
        try:
            experiment = build_station_equity(
                trips_path=args.trips_file,
                station_hour_path=args.station_hour_file,
                coordinate_path=args.coordinate_file,
                evaluation_start=datetime.fromisoformat(args.evaluation_start),
                evaluation_end=datetime.fromisoformat(args.evaluation_end),
                station_csv_path=args.station_output,
                figure_path=args.figure_output,
                json_path=args.json_output,
                markdown_path=args.markdown_output,
            )
        except (SimulationError, ValueError, OSError) as exc:
            print(f"Station equity simulation failed: {exc}")
            return 1

        default = experiment.equity_summaries["greedy_default"]
        service = experiment.equity_summaries["greedy_service"]
        print(f"Station equity scope: {experiment.stations}")
        print(
            "P2 default improved/tied/worsened stations: "
            f"{default['stations_improved']}/"
            f"{default['stations_tied']}/"
            f"{default['stations_worsened']}"
        )
        print(
            "P2 service improved/tied/worsened stations: "
            f"{service['stations_improved']}/"
            f"{service['stations_tied']}/"
            f"{service['stations_worsened']}"
        )
        print(f"Report: {args.markdown_output}")
        return 0
    if args.command == "run-harm-trace":
        try:
            experiment = build_harm_trace(
                trips_path=args.trips_file,
                station_hour_path=args.station_hour_file,
                coordinate_path=args.coordinate_file,
                evaluation_start=datetime.fromisoformat(args.evaluation_start),
                evaluation_end=datetime.fromisoformat(args.evaluation_end),
                harm_csv_path=args.harm_output,
                figure_path=args.figure_output,
                json_path=args.json_output,
                markdown_path=args.markdown_output,
            )
        except (SimulationError, ValueError, OSError) as exc:
            print(f"Harm trace simulation failed: {exc}")
            return 1

        default = experiment.transition_summaries["greedy_default"]
        service = experiment.transition_summaries["greedy_service"]
        print(
            "P2 default rescued/harmed/net: "
            f"{default['rescued_requests']}/"
            f"{default['harmed_requests']}/"
            f"{default['net_failures_avoided']}"
        )
        print(
            "P2 service rescued/harmed/net: "
            f"{service['rescued_requests']}/"
            f"{service['harmed_requests']}/"
            f"{service['net_failures_avoided']}"
        )
        print(f"Report: {args.markdown_output}")
        return 0
    if args.command == "run-donor-reserve-training":
        try:
            experiment = build_donor_reserve_training(
                trips_path=args.trips_file,
                station_hour_path=args.station_hour_file,
                coordinate_path=args.coordinate_file,
                training_start=datetime.fromisoformat(args.training_start),
                training_end=datetime.fromisoformat(args.training_end),
                reserves=tuple(args.reserves),
                comparison_csv_path=args.comparison_output,
                figure_path=args.figure_output,
                json_path=args.json_output,
                markdown_path=args.markdown_output,
            )
        except (SimulationError, ValueError, OSError) as exc:
            print(f"Donor reserve training failed: {exc}")
            return 1

        print(f"Pareto donor reserves: {list(experiment.pareto_reserves)}")
        print(f"Selection status: {experiment.selection_status}")
        print(f"Report: {args.markdown_output}")
        return 0
    if args.command == "run-donor-reserve-holdout":
        try:
            experiment = build_donor_reserve_holdout(
                trips_path=args.trips_file,
                station_hour_path=args.station_hour_file,
                coordinate_path=args.coordinate_file,
                evaluation_start=datetime.fromisoformat(args.evaluation_start),
                evaluation_end=datetime.fromisoformat(args.evaluation_end),
                selected_reserve=args.selected_reserve,
                comparison_csv_path=args.comparison_output,
                figure_path=args.figure_output,
                json_path=args.json_output,
                markdown_path=args.markdown_output,
            )
        except (SimulationError, ValueError, OSError) as exc:
            print(f"Donor reserve holdout failed: {exc}")
            return 1

        comparison = experiment.holdout_comparison
        print(f"Frozen donor reserve: {experiment.selected_reserve}")
        print(f"Holdout pattern: {comparison['pattern_on_frozen_pareto_axes']}")
        print(
            "Reserve change failed/harmed/p10pp: "
            f"{comparison['failed_rentals_change']:+}/"
            f"{comparison['harmed_requests_change']:+}/"
            f"{comparison['p10_service_rate_pp_change']:+.3f}"
        )
        print(f"Report: {args.markdown_output}")
        return 0
    if args.command == "run-fleet-sensitivity":
        try:
            experiment = build_fleet_sensitivity(
                trips_path=args.trips_file,
                station_hour_path=args.station_hour_file,
                coordinate_path=args.coordinate_file,
                evaluation_start=datetime.fromisoformat(args.evaluation_start),
                evaluation_end=datetime.fromisoformat(args.evaluation_end),
                fleet_sizes=tuple(args.fleet_sizes),
                comparison_csv_path=args.comparison_output,
                figure_path=args.figure_output,
                json_path=args.json_output,
                markdown_path=args.markdown_output,
            )
        except (SimulationError, ValueError, OSError) as exc:
            print(f"Fleet sensitivity failed: {exc}")
            return 1

        print(
            f"Fleet sensitivity: {experiment.stations} stations, "
            f"{experiment.observed_requests} requests"
        )
        for comparison in experiment.fleet_vs_instant:
            print(
                f"F{comparison['fleet_size']} failed/distance/utilization: "
                f"{comparison['failed_rentals_change']:+}/"
                f"{comparison['total_distance_km_change']:+.1f}km/"
                f"{comparison['fleet_utilization_rate']:.1%}"
            )
        print(f"Report: {args.markdown_output}")
        return 0
    if args.command == "run-simulation":
        try:
            experiment = build_policy_comparison(
                trips_path=args.trips_file,
                station_hour_path=args.station_hour_file,
                training_start=datetime.fromisoformat(args.training_start),
                training_end=datetime.fromisoformat(args.training_end),
                evaluation_start=datetime.fromisoformat(args.evaluation_start),
                evaluation_end=datetime.fromisoformat(args.evaluation_end),
                decision_interval_minutes=args.decision_interval_minutes,
                max_bikes_per_decision=args.max_bikes_per_decision,
                training_csv_path=args.training_output,
                comparison_csv_path=args.comparison_output,
                station_csv_path=args.station_output,
                figure_path=args.figure_output,
                json_path=args.json_output,
                markdown_path=args.markdown_output,
            )
        except (SimulationError, ValueError, OSError) as exc:
            print(f"Simulation failed: {exc}")
            return 1

        print(f"Selected threshold candidate: {experiment.selected_candidate['label']}")
        print(f"Evaluation failures avoided: {experiment.improvement['failures_avoided']:,.0f}")
        print(
            "Evaluation empty hours reduced: "
            f"{experiment.improvement['empty_station_hours_reduced']:,.1f}"
        )
        print(f"Report: {args.markdown_output}")
        return 0
    return 2
