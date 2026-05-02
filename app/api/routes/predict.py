import cv2
import numpy as np
import base64
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.gesture import process_external_frame

router = APIRouter(tags=["AI Prediction"])

class FrameData(BaseModel):
    image: str  # Base64 string dari browser

@router.post("/predict")
async def predict_frame(data: FrameData):
    """
    Menerima frame gambar base64 dari browser, 
    memprosesnya dengan AI, dan mengembalikan hasil deteksi.
    """
    try:
        # Decode base64 ke OpenCV image
        header, encoded = data.image.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Gagal decode gambar")

        # Proses frame dengan logic gesture yang sama
        result = process_external_frame(frame)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
