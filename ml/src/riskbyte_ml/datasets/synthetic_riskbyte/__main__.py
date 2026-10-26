"""Explicit deterministic synthetic generator and optional Phase 2 offline pipeline."""

import argparse
import json
from pathlib import Path

from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.config import (
    GeneratorConfig,
    SyntheticConfig,
    load_generator_config,
)
from riskbyte_ml.datasets.synthetic_riskbyte.generator import generate, save_generated


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="IEEE-CIS-inspired synthetic financial transaction dataset"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--preset", choices=["small", "medium", "large"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("ml/data/synthetic_riskbyte"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("ml/artifacts"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-pipeline", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_generator_config(args.config) if args.config else GeneratorConfig()
        changes = {k: getattr(args, k) for k in ("preset", "seed") if getattr(args, k) is not None}
        config = GeneratorConfig.model_validate({**config.model_dump(), **changes})
        manifest = save_generated(generate(config), args.output_dir, overwrite=args.overwrite)
        if args.run_pipeline:
            from riskbyte_ml.datasets.synthetic_riskbyte.pipeline import run_phase2

            directory = args.output_dir.resolve()
            run_phase2(
                SyntheticConfig(
                    transaction_path=directory / "train_transaction.csv",
                    identity_path=directory / "train_identity.csv",
                    metadata_path=directory / "generation_metadata.json",
                ),
                args.artifacts_dir,
            )
        print(
            json.dumps(
                {
                    "status": "synthetic_dataset_generated",
                    "dataset_type": "SYNTHETIC",
                    "fingerprint": manifest["fingerprint"],
                    "transaction_rows": manifest["transaction_count"],
                    "fraud_prevalence": manifest["fraud_prevalence"],
                }
            )
        )
        return 0
    except (DatasetError, ValueError, OSError) as error:
        print(
            json.dumps(
                {
                    "status": "dataset_configuration_error",
                    "error": {
                        "code": error.code
                        if isinstance(error, DatasetError)
                        else "synthetic_pipeline_failed",
                        "message": error.message
                        if isinstance(error, DatasetError)
                        else "Synthetic pipeline failed; inspect local configuration",
                    },
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
