from collections import Counter
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.db_models import DiagnosisLog, FieldAnalysisLog, CropRecommendationLog
from app.services.diagnosis_service import diagnosis_service
from app.services.field_service import field_service
from app.services.crop_recommender_service import crop_recommender_service
from app.services.ripeness_service import ripeness_service
from app.services.image_router_service import image_router_service
from app.services.disease_type_service import disease_type_service

router = APIRouter()


def _field_health(log: FieldAnalysisLog) -> int:
    # Same formula as FieldService.analyze
    return max(15, 100 - log.pest_count * 12 - log.weed_count * 10 - (30 if log.wilting_detected else 0))


@router.get("/stats")
def get_dashboard_stats(db: Session = Depends(get_db)):
    """
    Executive dashboard numbers computed from logged scans and real model metrics.
    """
    since = datetime.utcnow() - timedelta(days=6)
    diagnoses = db.query(DiagnosisLog).all()
    field_logs = db.query(FieldAnalysisLog).all()
    latest_crop = db.query(CropRecommendationLog).order_by(CropRecommendationLog.id.desc()).first()
    crop_counts = Counter(r.recommended_crop for r in db.query(CropRecommendationLog).all())

    trend = []
    for offset in range(6, -1, -1):
        day = (datetime.utcnow() - timedelta(days=offset)).date()
        day_diag = [d for d in diagnoses if d.created_at.date() == day]
        day_field = [f for f in field_logs if f.created_at.date() == day]
        trend.append({
            "day": day.strftime("%a"),
            "diagnoses": len(day_diag),
            "diseased": sum(1 for d in day_diag if d.cause_category != "none"),
            "fieldHealth": round(sum(map(_field_health, day_field)) / len(day_field)) if day_field else None,
        })

    recent_field = [f for f in field_logs if f.created_at >= since]
    return {
        "diagnoses_total": len(diagnoses),
        "diseased_total": sum(1 for d in diagnoses if d.cause_category != "none"),
        "field_scans_total": len(field_logs),
        "avg_field_health_7d": round(sum(map(_field_health, recent_field)) / len(recent_field)) if recent_field else None,
        "latest_crop_recommendation": latest_crop.recommended_crop if latest_crop else None,
        "top_recommended_crops": [{"crop": c, "count": n} for c, n in crop_counts.most_common(5)],
        "trend_7d": trend,
        "model_metrics": {
            "disease_classifier": diagnosis_service.metrics,
            "crop_recommender": crop_recommender_service.metrics,
            "ripeness_classifier": ripeness_service.metrics,
            "image_router": image_router_service.metrics,
            "disease_type_classifier": disease_type_service.metrics,
            **{f"{k}_classifier": v for k, v in field_service.metrics.items()},
        },
    }
