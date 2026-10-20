from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict
from sqlalchemy.orm import Session

from app.services.chatbot_service import chatbot_service
from app.db.session import get_db
from app.models.db_models import ChatMessageLog, DiagnosisLog

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: Optional[str] = Field("default_session", max_length=100)
    message: str = Field(..., max_length=2000)
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list, max_length=50)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message must not be empty")
        return v

@router.post("/message")
async def chat_message(req: ChatRequest, db: Session = Depends(get_db)):
    """
    Agricultural chatbot RAG / Q&A endpoint for plant care, disease prevention, and farming advice.
    """
    recent = db.query(DiagnosisLog).order_by(DiagnosisLog.id.desc()).limit(3).all()
    recent_scans = [{
        "id": d.id, "predicted_class": d.predicted_class, "confidence": d.confidence,
        "cause_category": d.cause_category, "pathogen_name": d.pathogen_name,
        "severity_score": d.severity_score, "remedies": d.remedies, "created_at": d.created_at,
    } for d in recent]
    result = chatbot_service.get_response(req.message, req.history, recent_scans)

    # Log chat to DB
    try:
        user_log = ChatMessageLog(session_id=req.session_id, role="user", content=req.message)
        bot_log = ChatMessageLog(session_id=req.session_id, role="assistant", content=result["bot_response"])
        db.add(user_log)
        db.add(bot_log)
        db.commit()
    except Exception as e:
        print(f"DB Logging Note: {e}")

    return result
