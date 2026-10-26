"""Lazy session dependency: importing the app never connects to PostgreSQL."""

from collections.abc import Iterator

from fastapi import Depends, Request
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from riskbyte_api.core.config import Settings


def create_database_engine(settings: Settings) -> Engine:
    if settings.database_url is None:
        raise RuntimeError("Database configuration is required for database-backed routes")
    return create_engine(settings.database_url.get_secret_value(), pool_pre_ping=True, echo=False)


def get_engine(request: Request) -> Engine:
    # Engine initialization belongs in the lifespan before DB-backed routes are introduced.
    engine = getattr(request.app.state, "database_engine", None)
    if engine is None:
        raise RuntimeError("Database engine has not been initialized")
    return engine


def get_session(engine: Engine = Depends(get_engine)) -> Iterator[Session]:
    with Session(engine) as session:
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        # Application services explicitly commit successful write transactions.
