"""Offline IEEE-CIS inspection entrypoint; no automatic downloads or preprocessing."""

import argparse
import json
from pathlib import Path

from riskbyte_ml.data.artifacts import write_artifact
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.ieee_cis.adapter import JoinRejected
from riskbyte_ml.datasets.ieee_cis.config import load_ieee_config
from riskbyte_ml.datasets.ieee_cis.pipeline import integrate, save_integration


def main(argv=None):
    parser = argparse.ArgumentParser(description="RiskByte IEEE-CIS Phase 1.5 inspection")
    parser.add_argument("--config", type=Path, default=Path("ml/config/ieee_cis.example.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("ml/artifacts/ieee_cis"))
    args = parser.parse_args(argv)
    try:
        config = load_ieee_config(args.config)
        profile, report = integrate(config)
        save_integration(profile, report, args.output_dir, config)
        print(
            json.dumps(
                {
                    "status": "dataset_ready"
                    if profile.quality.validation_status == "PASSED"
                    else "dataset_validation_failed",
                    "fingerprint": report.fingerprint,
                    "transaction_rows": report.join_audit.transaction_rows_before,
                    "identity_rows": report.join_audit.identity_rows_before,
                    "joined_rows": report.join_audit.joined_rows,
                    "fraud_prevalence": profile.quality.target.positive_class_prevalence,
                    "identity_coverage_percentage": report.join_audit.identity_coverage_percentage,
                    "memory_usage_bytes": report.memory.joined_after_bytes,
                }
            )
        )
        return 0 if profile.quality.validation_status == "PASSED" else 2
    except DatasetError as error:
        if isinstance(error, JoinRejected):
            audit_path = args.output_dir / "join_audit.json"
            if audit_path.resolve() not in {
                config.transaction_path.resolve(),
                config.identity_path.resolve(),
            }:
                try:
                    write_artifact(audit_path, error.audit.model_dump_json(indent=2).encode())
                except DatasetError:
                    pass  # Preserve the original join failure; never publish a joined profile.
        print(
            json.dumps(
                {
                    "status": "dataset_configuration_error",
                    "error": {"code": error.code, "message": error.message},
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
