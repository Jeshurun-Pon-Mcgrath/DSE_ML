"""Strict pre-outcome transaction input and bounded public prediction contracts."""

from typing import Literal

from pydantic import Field, FiniteFloat, JsonValue

from riskbyte_ml.data.models import ReportModel


class TransactionInput(ReportModel):
    TransactionID: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    TransactionDT: FiniteFloat = Field(ge=0, le=1e12)
    TransactionAmt: FiniteFloat = Field(ge=0, le=1e12)
    behavioural_entity_id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_-]+$")
    merchant_id: str | None = Field(default=None, max_length=100)
    merchant_category: str | None = Field(default=None, max_length=80)
    transaction_type: str | None = Field(default=None, max_length=80)
    product_code: str | None = Field(default=None, max_length=80)
    card_type: str | None = Field(default=None, max_length=80)
    card_network: str | None = Field(default=None, max_length=80)
    billing_region: str | None = Field(default=None, max_length=80)
    shipping_region: str | None = Field(default=None, max_length=80)
    P_emaildomain: str | None = Field(default=None, max_length=100)
    R_emaildomain: str | None = Field(default=None, max_length=100)
    failed_attempts_before: int | None = Field(default=None, ge=0, le=100000)
    account_age_days: FiniteFloat | None = Field(default=None, ge=0, le=100000)
    is_international: Literal[0, 1] | None = None
    is_card_present: Literal[0, 1] | None = None
    DeviceType: str | None = Field(default=None, max_length=80)
    DeviceInfo: str | None = Field(default=None, max_length=100)
    device_id: str | None = Field(default=None, max_length=100)
    os_family: str | None = Field(default=None, max_length=80)
    browser_family: str | None = Field(default=None, max_length=80)
    identity_region: str | None = Field(default=None, max_length=80)
    identity_age_days: FiniteFloat | None = Field(default=None, ge=0, le=100000)


class BatchInput(ReportModel):
    transactions: list[TransactionInput] = Field(min_length=1, max_length=500)


class Risk(ReportModel):
    score: FiniteFloat = Field(ge=0, le=1)
    level: Literal["LOW", "GUARDED", "ELEVATED", "HIGH", "CRITICAL"]
    investigation_required: bool


class PredictionResponse(ReportModel):
    transaction_id: str
    risk: Risk
    supervised: dict[str, JsonValue]
    anomaly: dict[str, JsonValue]
    behaviour: dict[str, JsonValue]
    abrf: dict[str, JsonValue]
    explanation: dict[str, JsonValue]
    versions: dict[str, str]


class InvestigationPatch(ReportModel):
    state: Literal["UNREVIEWED", "UNDER_REVIEW", "CLEARED", "ESCALATED"]


class TransactionPage(ReportModel):
    total: int
    page: int
    page_size: int
    items: list[dict[str, JsonValue]]
