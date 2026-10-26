"""Initial runtime transaction, immutable prediction, review and audit tables."""

import sqlalchemy as sa
from alembic import op

revision = "20261014_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "transactions",
        sa.Column("id", sa.String(100), primary_key=True),
        sa.Column("elapsed_seconds", sa.Float, nullable=False),
        sa.Column("entity", sa.String(100), nullable=False),
        sa.Column("context", sa.JSON, nullable=False),
        sa.Column("features", sa.JSON, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
    )
    op.create_table(
        "predictions",
        sa.Column(
            "transaction_id", sa.String(100), sa.ForeignKey("transactions.id"), primary_key=True
        ),
        sa.Column("risk_score", sa.Float, nullable=False),
        sa.Column("risk_level", sa.String(20), nullable=False),
        sa.Column("investigation_required", sa.Boolean, nullable=False),
        sa.Column("result", sa.JSON, nullable=False),
        sa.Column("explanation", sa.JSON, nullable=True),
    )
    op.create_table(
        "investigations",
        sa.Column(
            "transaction_id", sa.String(100), sa.ForeignKey("transactions.id"), primary_key=True
        ),
        sa.Column("state", sa.String(30), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
    )
    op.create_table(
        "audit_events",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "transaction_id", sa.String(100), sa.ForeignKey("transactions.id"), nullable=False
        ),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("safe_metadata", sa.JSON, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
    )
    for table, columns in {
        "transactions": ["elapsed_seconds", "entity"],
        "predictions": ["risk_score", "risk_level", "investigation_required"],
        "audit_events": ["transaction_id"],
    }.items():
        for column in columns:
            op.create_index(f"ix_{table}_{column}", table, [column])


def downgrade():
    for table in ["audit_events", "investigations", "predictions", "transactions"]:
        op.drop_table(table)
