import os

from sqlalchemy import create_engine
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


def get_db():
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
