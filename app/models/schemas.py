from pydantic import BaseModel

class TextRequest(BaseModel):
    content: str

class ImageUrlRequest(BaseModel):
    url: str