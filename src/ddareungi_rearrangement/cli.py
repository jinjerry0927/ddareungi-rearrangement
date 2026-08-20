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
from ddareungi_rearrangement.simulation import SimulationError, build_policy_comparison


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
