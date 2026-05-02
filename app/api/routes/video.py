from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.services.gesture import generate_frames

router = APIRouter(tags=["Video"])


@router.get("/video_feed")
def video_feed():
    """
    Live streaming video kamera dengan overlay hasil deteksi gestur.
    Format: MJPEG (multipart/x-mixed-replace).
    Buka di browser: http://<IP_RASPBERRY_PI>:8000/video_feed
    """
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
