from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict
from sqlalchemy.orm import Session

from app.services.chatbot_service import chatbot_service
from app.db.session import get_db
from app.models.db_models import ChatMessageLog, DiagnosisLog

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: Optional[str] = "default_session"
    message: str
    history: Optional[List[Dict[str, str]]] = []

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
