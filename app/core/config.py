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

# --- Default Gesture Mapping (Fallback) ---
# Format asli (dari Firebase): Kategori -> List Gestur
DEFAULT_GESTURE_CONFIG: dict = {
    "Makan": ["A"],
    "Minum": ["B"],
    "Toilet": ["C"],
    "Tidur": ["D"],
}

# Fungsi helper untuk membalik mapping agar mudah digunakan oleh AI (Gestur -> Kategori)
def generate_flat_mapping(config: dict) -> dict:
    flat = {}
    for kebutuhan, daftar_gestur in config.items():
        for gestur in daftar_gestur:
            flat[gestur] = kebutuhan
    return flat

# Mapping yang digunakan in-memory oleh AI
DEFAULT_GESTURE_MAPPING: dict = generate_flat_mapping(DEFAULT_GESTURE_CONFIG)
