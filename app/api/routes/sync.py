from fastapi import APIRouter
from app.core.firebase import firebase_queue

router = APIRouter(tags=["Firebase"])


@router.post("/sync")
def sync_firebase():
    """
    Memaksa sinkronisasi gesture mapping dari Firebase Firestore.
    Berguna jika konfigurasi gestur baru saja diubah dari sisi admin/Flutter.
    """
    firebase_queue.put({"action": "sync_config"})
    return {"status": "Sync request sent to Firebase worker"}
