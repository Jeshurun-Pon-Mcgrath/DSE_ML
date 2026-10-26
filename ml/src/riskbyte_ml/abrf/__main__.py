"""Run frozen-score ABRF research; there is no test or future-holdout option."""

import argparse
import json
from pathlib import Path

from riskbyte_ml.abrf.contracts import SearchConfig
from riskbyte_ml.abrf.pipeline import run
from riskbyte_ml.abrf.report import write_reports
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.config import load_synthetic_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--search-config", type=Path)
    parser.add_argument("--write-reports", action="store_true")
    args = parser.parse_args()
    try:
        config = load_synthetic_config(args.config)
        search = (
            SearchConfig.model_validate_json(args.search_config.read_bytes())
            if args.search_config
            else SearchConfig()
        )
        root = Path(__file__).resolve().parents[3] / "artifacts"
        new = not (root / "abrf" / "active.json").exists()
        bundle, directory = run(config, root, search)
        if new or args.write_reports:
            write_reports(bundle, root.parents[1] / "docs")
        print(
            json.dumps(
                {
                    "selected": bundle.selected.model_dump(mode="json"),
                    "validation": bundle.validation,
                    "improvement": bundle.improvement,
                    "search_summary": bundle.search_summary,
                    "future_holdout_status": bundle.future_holdout_status,
                    "original_temporal_test_accessed": False,
                },
                indent=2,
                allow_nan=False,
            )
        )
    except (DatasetError, ValueError, OSError) as exc:
        print(
            f"ABRF aborted: {getattr(exc, 'message', 'Invalid/unavailable fusion configuration')}"
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
