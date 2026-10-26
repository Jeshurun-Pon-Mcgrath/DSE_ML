"""Defensible encoded-to-source mapping, never inferred business meanings for V features."""

from riskbyte_ml.explainability.schemas import FeatureExplanation

EXCLUDED = {
    "isFraud",
    "fraud_scenario",
    "hard_negative",
    "hard_negative_type",
    "TransactionID",
    "behavioural_entity_id",
}


def display_name(name, parent):
    labels = {
        "TransactionAmt": "Transaction amount",
        "amount_ratio_to_prior_mean": "Amount relative to strictly prior mean",
        "amount_zscore_prior": "Amount deviation from strictly prior mean in standard deviations",
        "amount_minus_prior_mean": "Amount minus strictly prior mean",
        "time_since_previous_transaction": "Elapsed seconds since previous transaction",
        "failed_attempts_before": "Failed attempts before this transaction",
        "device_seen_before": "Device previously observed in entity history",
        "merchant_seen_before": "Merchant previously observed in entity history",
        "billing_region_seen_before": "Billing region previously observed in entity history",
        "behavioural_history_count": "Strictly prior behavioural history count",
    }
    label = labels.get(parent, parent.replace("_", " "))
    if parent.startswith("transactions_prior_"):
        label = "Strictly prior transaction count within " + parent.removeprefix(
            "transactions_prior_"
        )
    if name.endswith("__missing"):
        return "Missing-value indicator for " + label
    if name.endswith("__train_frequency"):
        return "Training-frequency encoding of " + label
    if name != parent:
        return label + " encoded category: " + name[len(parent) :].lstrip("_")
    return label


def family(name):
    if name.startswith("V") and name[1:].isdigit():
        return "anonymized_engineered"
    if name == "TransactionAmt":
        return "transaction_amount"
    if name.startswith("transactions_prior_"):
        return "velocity"
    if "amount" in name:
        return "behavioural_amount_history"
    if "device" in name.lower() or name in {"os_family", "browser_family"}:
        return "device"
    if "merchant" in name:
        return "merchant"
    if "region" in name or name == "is_international":
        return "region"
    if "failed_attempt" in name:
        return "failed_attempts"
    if "product" in name:
        return "product"
    if "identity" in name:
        return "identity"
    if name in {"elapsed_day_index", "relative_time_position", "time_since_previous_transaction"}:
        return "relative_time"
    if "card" in name or "transaction_type" in name:
        return "payment_context"
    if "emaildomain" in name:
        return "email_context"
    return "account_history"


def encoded_metadata(manifest, names):
    raw = {s["name"]: s for s in manifest["features"]}
    encoded = manifest["families"]["tree_boosting"]["feature_manifest"]
    if names != [s["name"] for s in encoded]:
        raise ValueError("Encoded feature order differs from frozen manifest")
    result = []
    for spec in encoded:
        name = spec["name"]
        parent = (
            name
            if name in raw
            else next(
                (n for n in sorted(raw, key=len, reverse=True) if name.startswith(n + "_")), None
            )
        )
        if parent is None or parent in EXCLUDED:
            raise ValueError("Unmapped or forbidden predictive feature")
        source = raw[parent]
        original = spec["interpretability_class"]
        interpretation = "INTERPRETABLE" if original == "TEMPORAL" else original
        if interpretation not in {"INTERPRETABLE", "PARTIALLY_INTERPRETABLE", "ANONYMIZED"}:
            raise ValueError("Unsupported predictive interpretability class")
        result.append(
            {
                "feature": name,
                "source_feature": parent,
                "source_columns": spec["source_columns"],
                "feature_family": family(parent),
                "raw_or_derived": source["kind"],
                "interpretability": interpretation,
                "manifest_interpretability": original,
                "transformation": spec["transformation"],
                "display_name": display_name(name, parent),
            }
        )
    return result


def statement(metadata, increasing):
    verb = "increased" if increasing else "decreased"
    if metadata["interpretability"] == "ANONYMIZED":
        return (
            f"An anonymized engineered feature {metadata['source_feature']} {verb} "
            "the classifier output; business semantics are unavailable."
        )
    return f"{metadata['display_name']} {verb} the underlying classifier's fraud-class output."


def top_features(values, attribution, metadata, k=5):
    if not 1 <= k <= 20 or len(values) != len(attribution) or len(metadata) != len(values):
        raise ValueError("Aligned features and top-K between 1 and 20 required")
    result = []
    for positive in [True, False]:
        indexes = (
            [i for i, v in enumerate(attribution) if v > 0]
            if positive
            else [i for i, v in enumerate(attribution) if v < 0]
        )
        indexes.sort(key=lambda i: (-abs(attribution[i]), metadata[i]["feature"]))
        result.append(
            [
                FeatureExplanation(
                    **{
                        key: metadata[i][key]
                        for key in [
                            "feature",
                            "display_name",
                            "interpretability",
                            "manifest_interpretability",
                            "source_columns",
                            "feature_family",
                            "raw_or_derived",
                        ]
                    },
                    feature_value=float(values[i]),
                    shap_value=float(attribution[i]),
                    direction="increases_classifier_risk"
                    if positive
                    else "decreases_classifier_risk",
                    statement=statement(metadata[i], positive),
                )
                for i in indexes[:k]
            ]
        )
    return result
