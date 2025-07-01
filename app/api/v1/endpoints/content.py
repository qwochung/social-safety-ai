from fastapi import APIRouter, HTTPException
from app.models.schemas import TextRequest
from app.utils.helpers import clean_text, custom_response
from app.services.model_loader import tokenizer, model_bert, device
from app.core.violation_types import ViolationType
import torch
import logging
router = APIRouter()

@router.post("/predict/content")
async def predict_content(request: TextRequest):
    logging.info(request)
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    cleaned = clean_text(request.content)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model_bert(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

    return custom_response(
        code=200,
        message="Success",
        data={
            "violation_type": ViolationType.VIOLENCE,
            "violation_detected": prediction,       # 1 = TOXIC, 0 = NON-TOXIC
            "confidence": round(confidence, 4)
        }
    )
