from fastapi import APIRouter
from app.core.firebase import global_status

router = APIRouter(tags=["Status"])


@router.get("/status")
def get_status():
    """Mengembalikan status gestur yang terdeteksi saat ini."""
    return global_status
