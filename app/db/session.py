"""Consistent synchronous SQLAlchemy sessions."""

from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


def create_database(url: str) -> tuple[Engine, sessionmaker[Session]]:
    """Create a thread-safe SQLite engine and session factory."""
    parsed = make_url(url)
    if parsed.database and parsed.database != ":memory:":
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        url,
        connect_args={"check_same_thread": False},
        **({"poolclass": StaticPool} if parsed.database in (None, "", ":memory:") else {}),
    )
    return engine, sessionmaker(bind=engine, expire_on_commit=False)
