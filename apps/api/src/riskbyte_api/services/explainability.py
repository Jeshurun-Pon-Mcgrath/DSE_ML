"""Checksum-verified JSON only: no scoring, pickle execution or SHAP initialization in HTTP."""

from riskbyte_api.core.config import REPOSITORY_ROOT
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.config import load_synthetic_config
from riskbyte_ml.explainability.inputs import verify_selection
from riskbyte_ml.explainability.pipeline import verify_outputs
from riskbyte_ml.explainability.schemas import (
    ExplanationBundle,
    ExplanationResponse,
    UnifiedExplanation,
)
from riskbyte_ml.statistics.verification import read_json


class ExplainabilityService:
    def __init__(self, settings):
        self.settings = settings

    def response(self, section, transaction_id=None):
        settings = self.settings
        if settings.dataset_config_path is None:
            return ExplanationResponse(status="dataset_not_configured")
        if settings.dataset_adapter != "synthetic_riskbyte":
            return ExplanationResponse(
                status="explainability_unavailable", reason="Frozen synthetic source required"
            )
        config_path = settings.dataset_config_path
        config_path = config_path if config_path.is_absolute() else REPOSITORY_ROOT / config_path
        config = load_synthetic_config(config_path)
        directory = settings.explainability_artifacts_dir
        directory = directory if directory.is_absolute() else REPOSITORY_ROOT / directory
        # Dataset reports may live in an adapter-specific subdirectory. Frozen model,
        # split and fusion artifacts share the ABRF directory's parent instead.
        component_directory = settings.abrf_artifacts_dir
        component_directory = (
            component_directory
            if component_directory.is_absolute()
            else REPOSITORY_ROOT / component_directory
        )
        root = component_directory.parent
        _, _, provenance = verify_selection(config, root)
        path = directory / "analysis_bundle.json"
        if not path.exists():
            return ExplanationResponse(status="explainability_not_generated")
        try:
            if path.stat().st_size > 100_000:
                raise ValueError
            bundle = ExplanationBundle.model_validate_json(path.read_bytes())
            if any(bundle.provenance.get(k) != v for k, v in provenance.items()):
                raise ValueError
            verify_outputs(directory, bundle)
            reports = {
                name: read_json(directory / name, 10_000_000)
                for name in bundle.artifacts
                if name.endswith(".json")
            }
            cases = reports["case_studies.json"]
            errors = {
                "hard_negatives": reports["hard_negative_explanations.json"],
                "false_negatives": reports["false_negative_explanations.json"],
            }
            if section == "validation":
                records = [c for c in cases["cases"] if c["status"] == "AVAILABLE"]
                records += (
                    errors["hard_negatives"]["records"] + errors["false_negatives"]["records"]
                )
                record = next(
                    (r for r in records if r["explanation"]["transaction_id"] == transaction_id),
                    None,
                )
                if record is None:
                    raise DatasetError(
                        "validation_explanation_not_precomputed",
                        "No offline validation explanation exists for this event",
                        404,
                    )
                data = UnifiedExplanation.model_validate(record["explanation"])
            else:
                sections = {
                    "status": {
                        "version": bundle.version,
                        "shap_status": bundle.shap_status,
                        "shap_version": reports.get("explainer_metadata.json", {}).get(
                            "shap_library_version"
                        ),
                        "explainer_type": reports.get("explainer_metadata.json", {}).get(
                            "explainer_type"
                        ),
                        "explained_model": reports.get("explainer_metadata.json", {}).get(
                            "explained_model"
                        ),
                        "provenance": bundle.provenance,
                        "scope": "VALIDATION explanations",
                        "future_holdout_status": bundle.future_holdout_status,
                        "isolation_audit": bundle.isolation_audit,
                    },
                    "global": reports["global_feature_importance.json"],
                    "families": reports["feature_family_importance.json"],
                    "case-studies": cases,
                    "validation-errors": errors,
                }
                data = sections[section]
        except (OSError, ValueError, KeyError):
            raise DatasetError(
                "explanation_snapshot_invalid", "Published explanation snapshot is invalid"
            ) from None
        return ExplanationResponse(
            status="explainability_ready"
            if bundle.shap_status == "AVAILABLE"
            else "explainability_partial",
            data=data,
            reason="shap_dependency_unavailable" if bundle.shap_status != "AVAILABLE" else None,
        )
