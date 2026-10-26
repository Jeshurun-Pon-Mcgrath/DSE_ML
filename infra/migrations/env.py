"""Operator-run migrations; credentials come from validated settings, never this file."""

from alembic import context
from sqlalchemy import URL, create_engine

from riskbyte_api.core.config import Settings
from riskbyte_api.db import models  # noqa: F401
from riskbyte_api.db.base import Base

settings = Settings()
url = (
    settings.database_url.get_secret_value()
    if settings.database_url
    else URL.create("sqlite+pysqlite", database=str(settings.development_database_path))
)
engine = create_engine(url, echo=False)
with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata)
    with context.begin_transaction():
        context.run_migrations()
