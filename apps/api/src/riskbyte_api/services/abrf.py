"""Read-only checksum-verified fusion analysis. No model execution or dataset parsing."""

from riskbyte_api.core.config import REPOSITORY_ROOT
from riskbyte_ml.abrf.contracts import ABRFBundle, ABRFResponse
from riskbyte_ml.abrf.inputs import safe_child, verify_components
from riskbyte_ml.abrf.pipeline import verify_completed
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.config import load_synthetic_config
from riskbyte_ml.statistics.verification import file_hash, read_json


class ABRFService:
    def __init__(self, settings):
        self.settings = settings

    def response(self, section):
        settings = self.settings
        if settings.dataset_config_path is None:
            return ABRFResponse(status="dataset_not_configured")
        if settings.dataset_adapter != "synthetic_riskbyte":
            return ABRFResponse(
                status="abrf_unavailable", reason="Frozen synthetic source required"
            )
        config_path = settings.dataset_config_path
        config_path = config_path if config_path.is_absolute() else REPOSITORY_ROOT / config_path
        config = load_synthetic_config(config_path)
        directory = settings.abrf_artifacts_dir
        directory = directory if directory.is_absolute() else REPOSITORY_ROOT / directory
        _, _, provenance = verify_components(config, directory.parent)
        if not (directory / "active.json").exists():
            return ABRFResponse(status="abrf_not_generated")
        try:
            active = read_json(directory / "active.json", 1024)
            run_id = active["run_id"]
            if len(run_id) != 64 or not all(c in "0123456789abcdef" for c in run_id):
                raise ValueError
            run = safe_child(directory, run_id)
            path = run / "analysis_bundle.json"
            if path.stat().st_size > 5_000_000 or file_hash(path) != active["bundle_hash"]:
                raise ValueError
            bundle = ABRFBundle.model_validate_json(path.read_bytes())
            verify_completed(run, bundle, provenance)
        except (OSError, ValueError, KeyError):
            raise DatasetError(
                "abrf_snapshot_invalid", "Published fusion snapshot is invalid"
            ) from None
        sections = {
            "status": {
                "version": bundle.version,
                "dataset_fingerprint": provenance["dataset_fingerprint"],
                "scope": "TRAIN/VALIDATION only",
                "future_holdout_status": bundle.future_holdout_status,
            },
            "config": {
                "configuration": bundle.selected.model_dump(mode="json"),
                "provenance": bundle.provenance,
                "selection_rationale": bundle.selection_rationale,
            },
            "validation": {
                "metrics": bundle.validation,
                "improvement": bundle.improvement,
                "support_analysis": bundle.support_analysis,
            },
            "ablation": {
                "rows": bundle.ablation,
                "constraint_study": bundle.constraint_study,
                "confidence_study": bundle.confidence_study,
            },
            "scenarios": bundle.scenario_analysis,
            "hard-negatives": bundle.hard_negative_analysis,
        }
        return ABRFResponse(status="abrf_ready", data=sections[section])
