"""Compact transaction store; analyst state is separate from immutable model output."""

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from riskbyte_api.db.base import Base


def now():
    return datetime.now(UTC)


class Transaction(Base):
    __tablename__ = "transactions"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    elapsed_seconds: Mapped[float] = mapped_column(Float, index=True)
    entity: Mapped[str] = mapped_column(String(100), index=True)
    context: Mapped[dict] = mapped_column(JSON)
    features: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Prediction(Base):
    __tablename__ = "predictions"
    transaction_id: Mapped[str] = mapped_column(ForeignKey("transactions.id"), primary_key=True)
    risk_score: Mapped[float] = mapped_column(Float, index=True)
    risk_level: Mapped[str] = mapped_column(String(20), index=True)
    investigation_required: Mapped[bool] = mapped_column(Boolean, index=True)
    result: Mapped[dict] = mapped_column(JSON)
    explanation: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class Investigation(Base):
    __tablename__ = "investigations"
    transaction_id: Mapped[str] = mapped_column(ForeignKey("transactions.id"), primary_key=True)
    state: Mapped[str] = mapped_column(String(30), default="UNREVIEWED")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[str] = mapped_column(ForeignKey("transactions.id"), index=True)
    action: Mapped[str] = mapped_column(String(50))
    safe_metadata: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
