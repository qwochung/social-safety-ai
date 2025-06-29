import re
import nltk
from nltk.corpus import stopwords
from fastapi.responses import JSONResponse

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join([w for w in text.split() if w not in stop_words])

def custom_response(code: int, message: str, data=None):
    return JSONResponse(status_code=code, content={"code": code, "message": message, "data": data})
