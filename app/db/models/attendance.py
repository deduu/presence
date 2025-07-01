# models.py

from sqlalchemy import Column, DateTime, Integer, String, Float, ForeignKey, LargeBinary, Date
from sqlalchemy.orm import declarative_base, relationship
from datetime import date, datetime
from sqlalchemy.sql import func
import pytz

from app.db.base import Base

# Define GMT+7 timezone
gmt_plus_7 = pytz.timezone('Asia/Bangkok')

# People Table


class Person(Base):
    __tablename__ = 'people'

    person_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    date_of_birth = Column(Date, nullable=True)
    address = Column(String, nullable=True)
    contact_number = Column(String, nullable=True)
    image_path = Column(String, nullable=True)

    registered_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    faces = relationship("Face", back_populates="person")


# Update Face model to include person_id
class Face(Base):
    __tablename__ = 'faces'

    face_id = Column(Integer, primary_key=True, index=True)
    face_encoding = Column(LargeBinary, nullable=False)
    first_seen = Column(DateTime(timezone=True), nullable=False,
                        server_default=func.now()
                        )
    last_seen = Column(DateTime(timezone=True), nullable=False,
                       server_default=func.now()
                       )

    person_id = Column(Integer, ForeignKey('people.person_id'),
                       nullable=True)  # Allow anonymous faces

    # Relationships
    person = relationship("Person", back_populates="faces")
    image_records = relationship("ImageRecord", back_populates="face")


# Image Records Table
class ImageRecord(Base):
    __tablename__ = 'image_records'

    record_id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, nullable=False)
    face_id = Column(Integer, ForeignKey('faces.face_id'), nullable=False)
    detection_time = Column(DateTime(timezone=True), nullable=False)
    face_location = Column(String, nullable=True)

    image_width = Column(Integer, nullable=True)   # NEW
    image_height = Column(Integer, nullable=True)  # NEW

    batch_tag = Column(String, nullable=True)

    face = relationship("Face", back_populates="image_records")


# Image Counts Table
class ImageCount(Base):
    __tablename__ = 'image_counts'

    image_id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, unique=True, nullable=False)
    face_count = Column(Integer, nullable=False)
    processed_time = Column(DateTime(timezone=True), nullable=False)
