from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class EnvironmentReport:
    project_root: Path
    python_supported: bool
    missing_directories: tuple[str, ...]
    api_key_configured: bool

    @property
    def ready(self) -> bool:
        return self.python_supported and not self.missing_directories


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def inspect_environment(root: Path | None = None) -> EnvironmentReport:
    root = (root or project_root()).resolve()
    load_dotenv(root / ".env")

    required_directories = (
        "configs",
        "data/raw",
        "data/interim",
        "data/processed",
        "data/sample",
        "docs",
        "notebooks",
        "reports",
        "scripts",
        "tests",
    )
    missing = tuple(path for path in required_directories if not (root / path).is_dir())
    python_supported = (3, 11) <= sys.version_info[:2] < (3, 13)
    api_key_configured = bool(os.getenv("SEOUL_OPEN_DATA_API_KEY", "").strip())

    return EnvironmentReport(
        project_root=root,
        python_supported=python_supported,
        missing_directories=missing,
        api_key_configured=api_key_configured,
    )


def format_report(report: EnvironmentReport) -> str:
    directory_status = "PASS" if not report.missing_directories else "FAIL"
    api_status = "CONFIGURED" if report.api_key_configured else "NOT CONFIGURED"
    lines = [
        f"Project root: {report.project_root}",
        f"Python 3.11/3.12: {'PASS' if report.python_supported else 'FAIL'}",
        f"Required directories: {directory_status}",
        f"Seoul API key: {api_status}",
    ]
    if report.missing_directories:
        lines.append("Missing: " + ", ".join(report.missing_directories))
    return "\n".join(lines)
