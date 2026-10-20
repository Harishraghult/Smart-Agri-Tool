from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.db.session import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="farmer") # farmer, agronomist, admin
    created_at = Column(DateTime, default=datetime.utcnow)

    diagnoses = relationship("DiagnosisLog", back_populates="user")
    field_analyses = relationship("FieldAnalysisLog", back_populates="user")


class DiagnosisLog(Base):
    __tablename__ = "diagnosis_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    image_filename = Column(String, nullable=False)
    crop_type = Column(String, nullable=True)
    predicted_class = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    cause_category = Column(String, nullable=False) # pathogen, pest, deficiency, none
    pathogen_name = Column(String, nullable=True)
    severity_score = Column(Float, nullable=True) # percentage
    remedies = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="diagnoses")


class FieldAnalysisLog(Base):
    __tablename__ = "field_analysis_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    location_name = Column(String, nullable=True)
    pest_count = Column(Integer, default=0)
    weed_count = Column(Integer, default=0)
    wilting_detected = Column(Integer, default=0) # 0 = healthy, 1 = wilted
    irrigation_alert = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="field_analyses")


class CropRecommendationLog(Base):
    __tablename__ = "crop_recommendation_logs"

    id = Column(Integer, primary_key=True, index=True)
    n = Column(Float, nullable=False)
    p = Column(Float, nullable=False)
    k = Column(Float, nullable=False)
    temperature = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    ph = Column(Float, nullable=False)
    rainfall = Column(Float, nullable=False)
    recommended_crop = Column(String, nullable=False)
    top_alternatives = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatMessageLog(Base):
    __tablename__ = "chat_message_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False) # user or assistant
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
