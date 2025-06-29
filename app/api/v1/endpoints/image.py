from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.model_loader import model_image, device
from app.utils.transforms import transform
from app.utils.helpers import custom_response
from app.core.violation_types import ViolationType
from typing import List
from PIL import Image
import io
import torch
import torch.nn.functional as F
router = APIRouter()

@router.post("/predict/image")
async def predict_image(files: List[UploadFile] = File(...)):
    try:
        results = []

        for file in files:
            # Đọc và xử lý ảnh
            image = Image.open(io.BytesIO(await file.read())).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device)

            # Dự đoán
            with torch.no_grad():
                outputs = model_image(img_tensor)
                probs = F.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probs, 1)

            # Ghi kết quả từng ảnh
            results.append({
                "filename": file.filename,
                "violation_type": ViolationType.BLOOD,
                "violation_detected": int(predicted_class.item()),
                "confidence": float(confidence.item())
            })

        return custom_response(200, "Success", data=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")