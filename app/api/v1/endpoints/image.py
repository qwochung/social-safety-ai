from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.model_loader import model_image, device
from app.utils.transforms import transform
from app.utils.helpers import custom_response
from app.core.violation_types import ViolationType
from PIL import Image
import torch, io

router = APIRouter()

@router.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model_image(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, 1)

        return custom_response(200, "Success", {
            "filename": file.filename,
            "violation_type": ViolationType.BLOOD,
            "violation_detected": int(predicted_class.item()),
            "confidence": float(confidence.item())
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
