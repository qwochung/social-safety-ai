from fastapi import APIRouter, HTTPException
from app.models.schemas import TextRequest
from app.utils.helpers import clean_text, custom_response
from app.services.model_loader import tokenizer, model_bert, device
from app.core.violation_types import ViolationType
import torch

router = APIRouter()

@router.post("/predict/content")
async def predict_content(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    cleaned = clean_text(request.text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model_bert(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    return custom_response(
        code=200,
        message="Success",
        data={
            "violation_type": ViolationType.VIOLENCE,
            "violation_detected": int(prediction.item()),  # 1 = TOXIC, 0 = NON-TOXIC
            "confidence": float(confidence.item()),
        }
    )
