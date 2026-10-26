"""Offline operator interface; no downloads, model fitting or automatic cleaning."""

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from riskbyte_ml.data.cleaning import apply_cleaning, save_processed
from riskbyte_ml.data.config import DatasetConfig, load_config
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.data.ingestion import load_csv
from riskbyte_ml.data.models import CleaningPlan
from riskbyte_ml.data.pipeline import profile_dataset, save_reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RiskByte Phase 1 data intelligence")
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser("profile", help="Inspect and save reports without cleaning")
    profile.add_argument("--config", type=Path)
    profile.add_argument("--output-dir", type=Path, default=Path("ml/artifacts"))
    clean = commands.add_parser("clean", help="Apply explicitly supplied source-bound operations")
    clean.add_argument("--config", type=Path, required=True)
    clean.add_argument("--plan", type=Path, required=True)
    clean.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config) if args.config else DatasetConfig()
        if config.source_path is None:
            print(json.dumps({"status": "dataset_not_configured"}))
            return 0
        if args.command == "profile":
            result = profile_dataset(config)
            save_reports(result, args.output_dir)
            status = (
                "dataset_ready"
                if result.quality.validation_status == "PASSED"
                else "dataset_validation_failed"
            )
            print(json.dumps({"status": status, "fingerprint": result.fingerprint}))
            return 0 if status == "dataset_ready" else 2
        try:
            with args.plan.open("rb") as stream:
                content = stream.read(1_000_001)
            if len(content) > 1_000_000:
                raise ValueError
            plan = CleaningPlan.model_validate_json(content)
        except (OSError, ValidationError, ValueError):
            raise DatasetError(
                "dataset_cleaning_plan_invalid", "Cleaning plan cannot be loaded"
            ) from None
        dataset = load_csv(config)
        frame, audit = apply_cleaning(dataset, plan)
        save_processed(dataset, frame, audit, args.output)
        print(
            json.dumps(
                {
                    "status": "cleaning_completed",
                    "processed_fingerprint": audit.processed_fingerprint,
                }
            )
        )
        return 0
    except DatasetError as exc:
        print(json.dumps({"error": {"code": exc.code, "message": exc.message}}))
        return 2
