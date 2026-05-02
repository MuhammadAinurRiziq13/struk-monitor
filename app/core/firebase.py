"""
Modul Firebase: inisialisasi koneksi dan worker thread untuk operasi Firestore.
Worker thread berjalan di background agar tidak memblokir loop AI utama.
"""
import os
import time
import queue
import threading
from datetime import datetime

from app.core.config import (
    USER_ID,
    COOLDOWN_SECONDS,
    FIREBASE_CRED_PATH,
    DEFAULT_GESTURE_MAPPING,
)

# ─── State ────────────────────────────────────────────────────────────────────
firebase_queue: queue.Queue = queue.Queue()
gesture_mapping: dict = dict(DEFAULT_GESTURE_MAPPING)
last_signal_time: dict = {}

# Status global yang dikembalikan oleh endpoint /status
global_status: dict = {
    "current_label": "Waiting...",
    "confidence": 0.0,
    "last_signal_sent": None,
}

# ─── Inisialisasi Firebase ────────────────────────────────────────────────────
db = None

try:
    import firebase_admin
    from firebase_admin import credentials, firestore as fs

    if os.path.exists(FIREBASE_CRED_PATH):
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
        db = fs.client()
        print("[OK] Firebase Terhubung!")
    else:
        print(f"[WARNING] Berjalan OFFLINE - credentials tidak ditemukan di '{FIREBASE_CRED_PATH}'")
except ImportError:
    print("[WARNING] firebase-admin belum diinstal. Berjalan OFFLINE.")


# ─── Firebase Worker Thread ───────────────────────────────────────────────────
def firebase_worker() -> None:
    """
    Worker thread yang memproses task Firebase secara asynchronous.
    Task yang didukung:
      - 'sync_config'  : Menarik gesture mapping terbaru dari Firestore.
      - 'push_history' : Menyimpan riwayat sinyal gestur ke Firestore.
    """
    global gesture_mapping

    while True:
        task = firebase_queue.get()
        if task is None:
            break

        action = task.get("action")

        # ── Sync konfigurasi gesture dari Firestore ──────────────────────────
        if action == "sync_config" and db:
            try:
                doc = (
                    db.collection("users")
                    .document(USER_ID)
                    .collection("config")
                    .document("gestures")
                    .get()
                )
                if doc.exists:
                    new_mapping: dict = {}
                    for kebutuhan, daftar_gestur in doc.to_dict().items():
                        for gestur in daftar_gestur:
                            new_mapping[gestur] = kebutuhan
                    gesture_mapping = new_mapping
                    print(f"[SYNC] Firebase gesture mapping: {gesture_mapping}")
            except Exception as e:
                print(f"[ERROR] Gagal sync config: {e}")

        # ── Push riwayat sinyal ke Firestore ─────────────────────────────────
        elif action == "push_history":
            kebutuhan: str = task.get("kebutuhan", "")
            gestur: str = task.get("gestur", "")
            current_time = time.time()

            cooldown_ok = (
                kebutuhan not in last_signal_time
                or (current_time - last_signal_time[kebutuhan]) > COOLDOWN_SECONDS
            )

            if cooldown_ok:
                print(f"[SIGNAL] MENGIRIM: '{kebutuhan}' (Gestur: {gestur})")
                global_status["last_signal_sent"] = (
                    f"{kebutuhan} at {datetime.now().strftime('%H:%M:%S')}"
                )

                if db:
                    try:
                        from firebase_admin import firestore as fs_inner
                        db.collection("users").document(USER_ID).collection("history").document().set(
                            {
                                "kategori": kebutuhan,
                                "gestur_pemicu": gestur,
                                "waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "timestamp": fs_inner.SERVER_TIMESTAMP,
                                "status": "terkirim",
                            }
                        )
                        print("[OK] Tersimpan di Firestore!")
                        last_signal_time[kebutuhan] = current_time
                    except Exception as e:
                        print(f"[ERROR] Gagal mengirim ke Firestore: {e}")
                else:
                    last_signal_time[kebutuhan] = current_time


        firebase_queue.task_done()


def start_firebase_worker() -> None:
    """Menjalankan Firebase worker di background thread daemon."""
    threading.Thread(target=firebase_worker, daemon=True, name="firebase-worker").start()
    firebase_queue.put({"action": "sync_config"})
    print("[FIREBASE] Worker thread dimulai.")
