"""Reusable one-to-one transaction-preserving LEFT join."""

from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.data.values import missing_mask, percentage
from riskbyte_ml.datasets.join_models import JoinAudit


class JoinRejected(DatasetError):
    def __init__(self, code: str, message: str, audit: JoinAudit):
        super().__init__(code, message)
        self.audit = audit


def audited_left_join(transaction, identity, policy, *, key, target, required=(), prefix="dataset"):
    JOIN_KEY, TARGET = key, target
    required = set(required) | {JOIN_KEY, TARGET}
    if not required <= set(transaction.columns) or JOIN_KEY not in identity.columns:
        raise DatasetError(
            f"{prefix}_required_columns_missing", "Required Dataset roles are absent"
        )
    tx_missing = missing_mask(transaction[JOIN_KEY], policy)
    id_missing = missing_mask(identity[JOIN_KEY], policy)
    tx_keys, id_keys = transaction.loc[~tx_missing, JOIN_KEY], identity.loc[~id_missing, JOIN_KEY]
    audit = JoinAudit(
        status="REJECTED",
        transaction_rows_before=len(transaction),
        identity_rows_before=len(identity),
        joined_rows=None,
        unique_transaction_ids=int(tx_keys.nunique()),
        unique_identity_ids=int(id_keys.nunique()),
        duplicate_transaction_keys=int(tx_keys[tx_keys.duplicated(keep=False)].nunique()),
        duplicate_identity_keys=int(id_keys[id_keys.duplicated(keep=False)].nunique()),
        duplicate_transaction_extra_rows=int(tx_keys.duplicated().sum()),
        duplicate_identity_extra_rows=int(id_keys.duplicated().sum()),
        missing_transaction_keys=int(tx_missing.sum()),
        missing_identity_keys=int(id_missing.sum()),
        matched_identity_rows=None,
        unmatched_transaction_rows=None,
        unused_identity_rows=None,
        identity_coverage_percentage=None,
        columns_added=[],
        target_integrity_verified=False,
    )
    if audit.missing_transaction_keys or audit.missing_identity_keys:
        raise JoinRejected(
            f"{prefix}_missing_join_keys", "Dataset join keys contain missing values", audit
        )
    if audit.duplicate_transaction_keys or audit.duplicate_identity_keys:
        raise JoinRejected(
            f"{prefix}_duplicate_join_keys",
            "Dataset join keys must be unique in both tables",
            audit,
        )
    overlap = set(transaction.columns) & set(identity.columns) - {JOIN_KEY}
    if overlap:
        raise JoinRejected(
            f"{prefix}_overlapping_columns",
            "Identity columns conflict with transaction columns",
            audit,
        )
    # Membership includes identity records whose optional attributes are all missing.
    matched = int(tx_keys.isin(id_keys).sum())
    unused = int((~id_keys.isin(tx_keys)).sum())
    joined = transaction.merge(identity, on=JOIN_KEY, how="left", sort=False, validate="one_to_one")
    if len(joined) != len(transaction) or not joined[JOIN_KEY].equals(transaction[JOIN_KEY]):
        raise JoinRejected(
            f"{prefix}_join_integrity", "Join changed transaction count or order", audit
        )
    if not joined[TARGET].equals(transaction[TARGET]):
        raise JoinRejected(
            f"{prefix}_target_integrity", "Join changed the transaction target", audit
        )
    audit = JoinAudit.model_validate(
        {
            **audit.model_dump(),
            "status": "JOINED",
            "joined_rows": len(joined),
            "matched_identity_rows": matched,
            "unmatched_transaction_rows": len(transaction) - matched,
            "unused_identity_rows": unused,
            "identity_coverage_percentage": percentage(matched, len(transaction)),
            "columns_added": [name for name in identity.columns if name != JOIN_KEY],
            "target_integrity_verified": True,
        }
    )
    return joined, audit
