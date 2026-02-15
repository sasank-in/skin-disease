from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func

from .db import Base


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    disease = Column(String, nullable=False)
    rating = Column(Integer, nullable=False)
    comments = Column(Text, nullable=True)
    email = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
