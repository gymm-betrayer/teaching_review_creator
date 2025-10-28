from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.sql import func
from .database import Base

class ArticleRequest(Base):
    __tablename__ = "article_requests"

    id = Column(Integer, primary_key=True, index=True)
    topic = Column(String, index=True)
    length = Column(Integer, default=500)
    status = Column(String, default="pending")  # pending, processing, success, failed
    pdf_path = Column(String, nullable=True)
    content = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())