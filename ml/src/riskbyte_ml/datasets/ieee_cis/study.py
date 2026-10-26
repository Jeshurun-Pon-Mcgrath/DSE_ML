"Descriptive proxy-entity study and reviewable feature semantics; no history fitting."

import re

from riskbyte_ml.data.values import missing_mask, numeric_view, percentage
from riskbyte_ml.datasets.ieee_cis.config import (
    AMOUNT,
    ELAPSED,
    JOIN_KEY,
    TARGET,
    IEEEConfig,
    is_anonymized_engineered,
)
from riskbyte_ml.datasets.ieee_cis.models import (
    BehaviouralFeasibility,
    ContractFeature,
    EntityStudy,
    FeatureContract,
    FeatureMetadata,
)


def feature_catalog(dataset, profile):
    excluded = {
        warning.column for warning in profile.warnings if warning.code.startswith("leakage_")
    }
    result = []
    for column in profile.schema_metadata.columns:
        name = column.name
        if name == TARGET:
            tags, explanation = ["TARGET"], "Configured fraud outcome; never a predictive input"
        elif name == JOIN_KEY:
            tags, explanation = (
                ["IDENTIFIER", "EXCLUDED_CANDIDATE"],
                "Join/audit key; direct prediction risks ordering artifacts",
            )
        elif name == ELAPSED:
            tags, explanation = (
                ["TEMPORAL", "EXCLUDED_CANDIDATE"],
                "Elapsed seconds; chronology and relative features only",
            )
        elif is_anonymized_engineered(name):
            tags, explanation = (
                ["ANONYMIZED"],
                "V1–V339 anonymized engineered feature; business meaning unavailable",
            )
        elif name in {AMOUNT, "DeviceType", "DeviceInfo", "P_emaildomain", "R_emaildomain"}:
            tags, explanation = (
                ["INTERPRETABLE"],
                (
                    "Readable amount/device/email-domain context; values do not "
                    "establish identity or causation"
                ),
            )
        elif name == "ProductCD" or re.fullmatch(r"(?:card[1-6]|addr[12]|dist[12]|[CDM]\d+)", name):
            tags, explanation = (
                ["PARTIALLY_INTERPRETABLE"],
                (
                    "Known family/context, but anonymized codes or detailed "
                    "construction require provenance review"
                ),
            )
        else:
            tags, explanation = (
                ["ANONYMIZED"],
                "Meaning not established from schema/name alone; inspect source documentation",
            )
        if re.fullmatch(r"card[1-6]|addr[12]", name) or name in {"P_emaildomain", "DeviceInfo"}:
            tags.append("POTENTIAL_BEHAVIOURAL_KEY")
        if name in excluded and "EXCLUDED_CANDIDATE" not in tags:
            tags.append("EXCLUDED_CANDIDATE")
            explanation += "; flagged for leakage review, not dropped"
        result.append(
            FeatureMetadata(
                column=name,
                classifications=tags,
                explanation=explanation,
                observed_dtype=column.storage_dtype,
                missing_percentage=profile.quality.missing_values[name].percentage,
                unique_count=column.unique_count,
            )
        )
    return result


def entity_key_study(dataset, config: IEEEConfig) -> list[EntityStudy]:
    frame, policy = dataset.frame, dataset.config
    result = []
    for index, fields in enumerate(config.entity_candidates, 1):
        missing = {
            field: int(missing_mask(frame[field], policy).sum()) if field in frame else len(frame)
            for field in fields
        }
        absent = [field for field in fields if field not in frame]
        complete = (
            None
            if absent
            else ~frame[list(fields)].apply(lambda series: missing_mask(series, policy)).any(axis=1)
        )
        eligible = 0 if complete is None else int(complete.sum())
        study = EntityStudy(
            name=f"behavioural_entity_candidate_{index}",
            fields=list(fields),
            status="UNAVAILABLE_WITH_REASON",
            reason="required_fields_absent" if absent else "no_complete_proxy_keys",
            eligible_rows=eligible,
            missing_rows=len(frame) - eligible,
            missing_percentage=percentage(len(frame) - eligible, len(frame)),
            missing_by_field=missing,
            unique_entities=None,
            median_transactions_per_entity=None,
            mean_transactions_per_entity=None,
            maximum_transactions=None,
            singleton_entity_percentage=None,
            repeated_history_row_percentage=None,
            rows_with_strictly_prior_elapsed_time_percentage=None,
            proxy_variation_entity_percentage={},
            statistically_useful=False,
            collision_risk=(
                "Unknown true-identity collision rate; shared "
                "cards/addresses/devices may merge unrelated actors"
            ),
            interpretation_limitation=(
                "Exact tuple of observed proxy strings, not a person/customer; "
                "missing components excluded, changes fragment history"
            ),
        )
        if eligible:
            grouped = frame.loc[
                complete,
                list(
                    dict.fromkeys(
                        (
                            *fields,
                            *(
                                probe
                                for probe in ("DeviceInfo", "addr1", "P_emaildomain")
                                if probe in frame
                            ),
                        )
                    )
                ),
            ].groupby(list(fields), observed=True, sort=False, dropna=False)
            counts = grouped.size()
            entities = len(counts)
            singleton = percentage(int(counts.eq(1).sum()), entities)
            repeated_rows = percentage(int(counts[counts.gt(1)].sum()), eligible)
            prior_percentage = None
            if ELAPSED in frame:
                elapsed, _, _, _ = numeric_view(frame[ELAPSED], policy)
                temporal = frame.loc[complete, list(fields)].copy()
                temporal[ELAPSED] = elapsed[complete].where(elapsed[complete].ge(0))
                earliest = temporal.groupby(list(fields), observed=True, sort=False)[
                    ELAPSED
                ].transform("min")
                prior_percentage = percentage(int(temporal[ELAPSED].gt(earliest).sum()), eligible)
            variation = {}
            for probe in ("DeviceInfo", "addr1", "P_emaildomain"):
                if probe in frame and probe not in fields:
                    # Treat blank probes as missing, never as an extra observed identity attribute.
                    probes = frame.loc[complete, [*fields, probe]].copy()
                    probes[probe] = probes[probe].where(~missing_mask(probes[probe], policy))
                    sizes = probes.groupby(list(fields), observed=True, sort=False)[probe].nunique()
                    variation[probe] = percentage(int(sizes.gt(1).sum()), entities)
            useful = (
                entities >= config.min_entity_count_for_recommendation
                and 100 - singleton >= config.min_repeated_entity_percentage
                and percentage(eligible, len(frame)) >= config.min_complete_row_percentage
            )
            study = EntityStudy.model_validate(
                {
                    **study.model_dump(),
                    "status": "AVAILABLE",
                    "reason": None,
                    "unique_entities": entities,
                    "median_transactions_per_entity": float(counts.median()),
                    "mean_transactions_per_entity": float(counts.mean()),
                    "maximum_transactions": int(counts.max()),
                    "singleton_entity_percentage": singleton,
                    "repeated_history_row_percentage": repeated_rows,
                    "rows_with_strictly_prior_elapsed_time_percentage": prior_percentage,
                    "proxy_variation_entity_percentage": variation,
                    "statistically_useful": useful,
                }
            )
        result.append(study)
    return result


def behavioural_feasibility(dataset, studies) -> list[BehaviouralFeasibility]:
    frame, policy = dataset.frame, dataset.config

    def observed(field):
        if field not in frame:
            return False
        if field in {AMOUNT, ELAPSED}:
            finite, _, invalid, infinite = numeric_view(frame[field], policy)
            return (
                bool(finite.notna().any())
                and not (invalid or infinite)
                and (field != ELAPSED or not bool(finite.lt(0).any()))
            )
        return bool((~missing_mask(frame[field], policy)).any())

    history_candidates = [
        study
        for study in studies
        if study.statistically_useful
        and (study.rows_with_strictly_prior_elapsed_time_percentage or 0) > 0
    ]
    definitions = (
        (
            "amount_deviation",
            [AMOUNT, ELAPSED],
            True,
            (
                "Past-only amount baseline under reviewed proxy; "
                "sparse/cold-start histories remain unavailable"
            ),
        ),
        (
            "transaction_velocity",
            [ELAPSED],
            True,
            "Counts in relative elapsed-time windows, not real customer activity",
        ),
        (
            "time_since_previous_transaction",
            [ELAPSED],
            True,
            (
                "Difference to strictly earlier elapsed event under a proxy; ties "
                "need explicit simultaneous-event policy"
            ),
        ),
        (
            "merchant_novelty",
            [],
            False,
            (
                "No documented merchant identifier; ProductCD/email domain cannot "
                "establish merchant novelty"
            ),
        ),
        (
            "device_novelty",
            ["DeviceInfo", ELAPSED],
            True,
            (
                "DeviceInfo string novelty in prior proxy history; incomplete "
                "identity coverage, not physical-device identity"
            ),
        ),
        (
            "location_address_novelty",
            ["addr1", "addr2", ELAPSED],
            True,
            "Novel anonymized address-code tuple only; not GPS/travel or verified residence",
        ),
        (
            "repeated_behavioural_history",
            [ELAPSED],
            True,
            "Repeated proxy group observations; no verified customer history",
        ),
        (
            "relative_elapsed_hour_day_index",
            [ELAPSED],
            False,
            "floor(seconds/3600) or floor(seconds/86400), never wall-clock hour or weekday",
        ),
        (
            "calendar_hour_weekday_month",
            [],
            False,
            "Undisclosed wall-clock reference; calendar conversion is unsupported",
        ),
        (
            "geographic_distance_impossible_travel",
            [],
            False,
            "No documented coordinates; dist fields have insufficient public semantics",
        ),
        (
            "verified_customer_history",
            [],
            False,
            "No documented clean customer_id; constructed proxies do not prove identity",
        ),
    )
    reports = []
    for name, fields, requires_entity, explanation in definitions:
        novelty_fields = (
            {"DeviceInfo"}
            if name == "device_novelty"
            else ({"addr1", "addr2"} if name == "location_address_novelty" else set())
        )
        history = any(not novelty_fields.intersection(study.fields) for study in history_candidates)
        prerequisites = bool(fields) and all(observed(field) for field in fields)
        support = "UNSUPPORTED"
        if prerequisites and (not requires_entity or history):
            support = "PARTIALLY_SUPPORTED" if requires_entity else "SUPPORTED"
        if requires_entity and not history:
            explanation += "; no statistically useful proxy established in this snapshot"
        reports.append(
            BehaviouralFeasibility(
                feature=name,
                support=support,
                required_fields=fields,
                requires_constructed_entity=requires_entity,
                explanation=explanation,
            )
        )
    return reports


def feature_contract(catalog, feasibility, profile) -> FeatureContract:
    exclusions = [
        item.column
        for item in catalog
        if "EXCLUDED_CANDIDATE" in item.classifications or "TARGET" in item.classifications
    ]
    categories = [
        item.column
        for item in catalog
        if item.column in profile.schema_metadata.categorical_columns
    ]
    engineered = [item.column for item in catalog if is_anonymized_engineered(item.column)]
    raw = [
        item.column
        for item in catalog
        if item.column not in {*exclusions, *categories, *engineered}
    ]
    temporal, behavioural = [], []
    for item in feasibility:
        proposed = ContractFeature(
            name=item.feature,
            source_fields=item.required_fields,
            support=item.support,
            rule=item.explanation,
        )
        if item.feature in {"relative_elapsed_hour_day_index", "calendar_hour_weekday_month"}:
            temporal.append(proposed)
        else:
            behavioural.append(proposed)
    return FeatureContract(
        version="ieee-cis-feature-contract-proposed-v1",
        raw_predictive_features=raw,
        temporal_derived_features=temporal,
        behavioural_derived_features=behavioural,
        missingness_indicators=[
            name for name, summary in profile.quality.missing_values.items() if summary.count
        ],
        categorical_features=[name for name in categories if name not in exclusions],
        anonymized_engineered_features=[name for name in engineered if name not in exclusions],
        identifier_excluded_fields=exclusions,
        target_field=TARGET,
        history_policy=(
            "Strictly prior elapsed events only; no future/full-dataset "
            "aggregates or target history; chronological splits; grouped "
            "sensitivity evaluation; tied events excluded from each other's "
            "history"
        ),
        preprocessing_policy=(
            "Proposed only: preserve missingness; fit imputation/encoding on "
            "training split only; no target-derived or identifier encodings "
            "without leakage review"
        ),
    )
