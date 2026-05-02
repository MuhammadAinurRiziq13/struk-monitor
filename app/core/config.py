"""
Konfigurasi aplikasi StrokeMonitor.
Semua nilai dibaca dari file .env di root project.
"""
import os
from dotenv import load_dotenv

# Load .env dari root project (dua level di atas file ini: app/core/ -> app/ -> root)
load_dotenv()

# --- Identitas Pasien ---
USER_ID: str = os.getenv("USER_ID", "patient_001")

# --- Timing ---
COOLDOWN_SECONDS: int = int(os.getenv("COOLDOWN_SECONDS", "60"))

# --- Path File ---
MODEL_PATH: str = os.getenv("MODEL_PATH", "models/model.tflite")
LABEL_PATH: str = os.getenv("LABEL_PATH", "models/labels.npy")
FIREBASE_CRED_PATH: str = os.getenv("FIREBASE_CRED_PATH", "config/firebase_credentials.json")

# --- Server ---
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

# --- Default Gesture Mapping (bisa di-override dari Firestore) ---
DEFAULT_GESTURE_MAPPING: dict = {
    "A": "Makan",
    "B": "Minum",
    "C": "Toilet",
    "D": "Tidur",
}
