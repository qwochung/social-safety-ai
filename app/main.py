from fastapi import FastAPI
from app.api.v1.routers import router as api_router
from app.core.exception_handlers import register_exception_handlers

app = FastAPI(title="Violation Detection API")

# Đăng ký routers
app.include_router(api_router, prefix="/api/v1")

# Đăng ký exception handlers
register_exception_handlers(app)








# import re
# from PIL import Image
# import nltk
# import torch
# from nltk.corpus import stopwords
# from fastapi import FastAPI, HTTPException, Request, UploadFile, File
# from fastapi.responses import JSONResponse
# from fastapi.exceptions import RequestValidationError
# from pydantic import BaseModel
# from torch import device
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#
# )
# import os
# import io
# from torchvision import models, transforms
# import torch.nn as nn
#
#
# # ====== Khởi tạo app và các tài nguyên ======
# app = FastAPI()
#
# # Tải stopwords nếu chưa có
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))
#
# def clean_text(text: str) -> str:
#     text = text.lower()
#     text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
#     text = re.sub(r"[^a-z\s]", "", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return " ".join([w for w in text.split() if w not in stop_words])
#
# # Response chuẩn
# def custom_response(code: int, message: str, data=None):
#     return JSONResponse(
#         status_code=code,
#         content={
#             "code": code,
#             "message": message,
#             "data": data
#         }
#     )
#
# # Model Input
# class TextRequest(BaseModel):
#     text: str
#
# # Load mô hình BERT
#
# LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
#
# model_bert = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
# tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
#
#
# # ====== Load mô hình hình ảnh ======
#
# # 1. Tạo lại kiến trúc ResNet18
# model_image = models.resnet18(pretrained=False)
# model_image.fc = nn.Linear(model_image.fc.in_features, 2)  # Số lớp là 2: blood vs no blood
#
# # 2. Load trọng số đã huấn luyện
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_image.load_state_dict(torch.load("../model/resnet18_blood_classify.pt", map_location=device))
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
#
# # 3. Đưa lên thiết bị và chuyển sang eval mode
# model_image.to(device)
# model_image.eval()
#
# # ====== Handler lỗi ======
# @app.exception_handler(HTTPException)
# async def http_exception_handler(request: Request, exc: HTTPException):
#     return custom_response(code=exc.status_code, message=exc.detail)
#
# @app.exception_handler(RequestValidationError)
# async def validation_exception_handler(request: Request, exc: RequestValidationError):
#     return custom_response(code=422, message="Invalid input", data=exc.errors())
#
# @app.exception_handler(Exception)
# async def global_exception_handler(request: Request, exc: Exception):
#     return custom_response(code=500, message=f"Internal Server Error: {str(exc)}")
#
#
# # ====== API: Dự đoán văn bản ======
# from pydantic import BaseModel
#
# class TextRequest(BaseModel):
#     text: str
#
# @app.post("/predict/content")
# async def predict_content(request: TextRequest):
#     if not request.text.strip():
#         raise HTTPException(status_code=400, detail="Text cannot be empty.")
#
#     cleaned_text = request.text.strip()
#     inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True).to(device)
#
#     with torch.no_grad():
#         outputs = model_bert(**inputs)
#         logits = outputs.logits
#         prediction = torch.argmax(logits, dim=1).item()
#
#     result = "TOXIC" if prediction == 1 else "NON-TOXIC"
#
#     return custom_response(
#         code=200,
#         message="Success",
#         data={"result": result}
#     )
#
#
#
# # ====== API: Dự đoán hình ảnh ======
# @app.post("/predict/image")
# async def predict_image(file: UploadFile = File(...)):
#     try:
#         image_bytes = await file.read()
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         img_tensor = transform(image).unsqueeze(0).to(device)
#
#         with torch.no_grad():
#             outputs = model_image(img_tensor)
#             probs = torch.nn.functional.softmax(outputs, dim=1)
#             confidence, predicted_class = torch.max(probs, 1)
#
#
#         return custom_response(
#             code=200,
#             message="Success",
#             data={
#                 "filename": file.filename,
#                 "violation_type": "blood",
#                 "violation_detected":  int(predicted_class.item()),
#                 "confidence": float(confidence.item())
#             }
#         )
#
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
