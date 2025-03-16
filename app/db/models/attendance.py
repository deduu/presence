# models.py

from sqlalchemy import Column, DateTime, Integer, String, Float, ForeignKey, LargeBinary
from sqlalchemy.orm import declarative_base, relationship
import datetime
import pytz

from app.db.base import Base

# Define GMT+7 timezone
gmt_plus_7 = pytz.timezone('Asia/Bangkok')

# Faces Table
class Face(Base):
    __tablename__ = 'faces'
    
    face_id = Column(Integer, primary_key=True, index=True)
    face_encoding = Column(LargeBinary, nullable=False)  # Stored as bytes
    first_seen = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.datetime.now(gmt_plus_7))
    last_seen = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.datetime.now(gmt_plus_7))
    
    # Relationships
    image_records = relationship("ImageRecord", back_populates="face")

# Image Records Table
class ImageRecord(Base):
    __tablename__ = 'image_records'
    
    record_id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, nullable=False)
    face_id = Column(Integer, ForeignKey('faces.face_id'), nullable=False)
    detection_time = Column(DateTime(timezone=True), nullable=False)
    
    # Relationships
    face = relationship("Face", back_populates="image_records")

# Image Counts Table
class ImageCount(Base):
    __tablename__ = 'image_counts'
    
    image_id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, unique=True, nullable=False)
    face_count = Column(Integer, nullable=False)
    processed_time = Column(DateTime(timezone=True), nullable=False)
