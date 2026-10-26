"""python -m riskbyte_ml.statistics; no synthetic data is generated here."""

import argparse
import json
from pathlib import Path

from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.config import load_synthetic_config
from riskbyte_ml.statistics.pipeline import generate


def main():
    parser = argparse.ArgumentParser(description="Frozen synthetic-data statistical intelligence")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("ml/artifacts"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = generate(
            load_synthetic_config(args.config),
            args.artifact_root,
            args.output or args.artifact_root / "statistics",
        )
    except DatasetError as exc:
        print(
            json.dumps(
                {"status": "analysis_aborted", "error": {"code": exc.code, "message": exc.message}}
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "statistics_ready",
                "dataset_fingerprint": result.provenance.dataset_fingerprint,
                "artifact_version": result.provenance.artifact_version,
                "hypotheses": result.hypotheses.family_size,
                "nominal_rejections": sum(
                    h.decision == "REJECT_H0_NOMINALLY" for h in result.hypotheses.hypotheses
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
