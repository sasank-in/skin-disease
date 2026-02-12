import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


def _db_url() -> str:
    db_path = os.getenv("DB_PATH", "app.db")
    return f"sqlite:///{db_path}"


def get_engine():
    return create_engine(_db_url(), connect_args={"check_same_thread": False})


def get_session_local():
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _ensure_columns(engine)


def _ensure_columns(engine) -> None:
    with engine.connect() as conn:
        try:
            result = conn.execute(text("PRAGMA table_info(users)"))
        except Exception:
            return
        existing = {row[1] for row in result.fetchall()}
        # Add missing columns for legacy databases
        if "first_name" not in existing:
            conn.execute(text("ALTER TABLE users ADD COLUMN first_name TEXT"))
        if "last_name" not in existing:
            conn.execute(text("ALTER TABLE users ADD COLUMN last_name TEXT"))
        if "phone" not in existing:
            conn.execute(text("ALTER TABLE users ADD COLUMN phone TEXT"))
        if "role" not in existing:
            conn.execute(text("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'"))
        conn.commit()


def get_db():
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
