"""
StrokeMonitor — Backend AI FastAPI
Entry point aplikasi. Jalankan dengan:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from app.core.config import HOST, PORT
from app.core.firebase import start_firebase_worker, stop_firebase_worker
from app.services.inference import load_model_and_labels
from app.services.gesture import init_gesture_service, shutdown_gesture_service
from app.api.routes import status, video, sync, predict


# ─── Lifespan: startup & shutdown ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler: semua inisialisasi berat dilakukan di sini agar
    FastAPI siap melayani request setelah semua komponen aktif.
    """
    print("=" * 50)
    print("  StrokeMonitor AI Backend - Starting Up")
    print("=" * 50)

    # 1. Muat model TFLite dan label
    interpreter, input_details, output_details, labels = load_model_and_labels()

    # 2. Inisialisasi kamera dan gesture service
    init_gesture_service(interpreter, input_details, output_details, labels)

    # 3. Jalankan Firebase worker thread
    start_firebase_worker()

    print("=" * 50)
    print(f"  [OK] Server siap di http://{HOST}:{PORT}")
    print(f"  [>>] Video feed: http://{HOST}:{PORT}/video_feed")
    print(f"  [i]  API docs : http://{HOST}:{PORT}/docs")
    print("=" * 50)

    yield  # Aplikasi berjalan di sini

    # Shutdown
    print("🛑 StrokeMonitor Backend shutting down...")
    stop_firebase_worker()   # unsubscribe listener + stop worker thread
    shutdown_gesture_service()


# ─── Aplikasi FastAPI ──────────────────────────────────────────────────────────
app = FastAPI(
    title="StrokeMonitor AI Backend",
    description=(
        "Backend AI untuk monitoring pasien stroke berbasis deteksi bahasa isyarat. "
        "Berjalan di Raspberry Pi, menggunakan MediaPipe + TFLite untuk klasifikasi gestur, "
        "dan Firebase Firestore untuk pengiriman sinyal ke aplikasi mobile via FCM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ─── Register Routes ───────────────────────────────────────────────────────────
app.include_router(status.router)
app.include_router(video.router)
app.include_router(sync.router)
app.include_router(predict.router)

# ─── Static Files & Frontend ──────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("app/static/index.html", "r", encoding="utf-8") as f:
        return f.read()


# ─── Entry point (opsional, jika dijalankan langsung) ─────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)
