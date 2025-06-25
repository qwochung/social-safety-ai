import re
import nltk
import torch
from nltk.corpus import stopwords
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# ====== Khởi tạo app và các tài nguyên ======
app = FastAPI()

# Tải stopwords nếu chưa có
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join([w for w in text.split() if w not in stop_words])

# Response chuẩn
def custom_response(code: int, message: str, data=None):
    return JSONResponse(
        status_code=code,
        content={
            "code": code,
            "message": message,
            "data": data
        }
    )

# Model Input
class TextRequest(BaseModel):
    text: str

# Load mô hình BERT

LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)


# ====== Handler lỗi ======
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return custom_response(code=exc.status_code, message=exc.detail)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return custom_response(code=422, message="Invalid input", data=exc.errors())

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return custom_response(code=500, message=f"Internal Server Error: {str(exc)}")

# ====== Endpoint chính ======
def predict_toxicity(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    cleaned_text = clean_text(request.text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    result = "TOXIC" if prediction == 1 else "NON-TOXIC"

    return custom_response(
        code=200,
        message="Success",
        data={"result": result}
    )
