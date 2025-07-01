from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.model_loader import model_image, device
from app.utils.transforms import transform
from app.utils.helpers import custom_response
from app.models.schemas import ImageUrlRequest


from PIL import Image
import io
import torch
import torch.nn.functional as F
import requests
from app.core.violation_types import ViolationType

router = APIRouter()

@router.post("/predict/image-url")
async def predict_image_from_url(request: ImageUrlRequest):
    try:
        # Tải ảnh từ URL (S3)
        response = requests.get(request.url)
        response.raise_for_status()

        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Dự đoán
        with torch.no_grad():
            outputs = model_image(img_tensor)
            probs = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()


        return custom_response(200, "Success", data={
            "violation_type": ViolationType.BLOOD,
            "violation_detected": predicted_class,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        print(f"Error fetching image: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")