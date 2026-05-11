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
_config_watch = None  # Referensi ke Firestore listener agar bisa di-unsubscribe

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

        # ── Push riwayat sinyal ke Firestore ─────────────────────────────────
        if action == "push_history":
            kebutuhan: str = task.get("kebutuhan", "")
            gestur: str = task.get("gestur", "")
            confidence: float = task.get("confidence", 0.0)
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
                        from firebase_admin import messaging
                        
                        db.collection("users").document(USER_ID).collection("notifications").document().set(
                            {
                                "category": kebutuhan,
                                "trigger_gesture": gestur,
                                "confidence": confidence,
                                "timestamp": fs_inner.SERVER_TIMESTAMP,
                                "notified_at": None,
                                "status": "pending",
                            }
                        )
                        print("[OK] Notifikasi tersimpan di Firestore!")
                        
                        # Kirim Push Notification FCM ke semua perangkat terdaftar
                        user_doc = db.collection("users").document(USER_ID).get()
                        if user_doc.exists:
                            user_data = user_doc.to_dict()
                            # Ambil list tokens (fcm_tokens) atau fallback ke single token (fcm_token)
                            fcm_tokens = user_data.get("fcm_tokens", [])
                            if not fcm_tokens and "fcm_token" in user_data:
                                fcm_tokens = [user_data["fcm_token"]]
                            
                            if fcm_tokens:
                                # Buat list pesan untuk setiap token
                                messages = [
                                    messaging.Message(
                                        notification=messaging.Notification(
                                            title="StrokeMonitor Alert",
                                            body=f"Pasien membutuhkan bantuan: {kebutuhan}"
                                        ),
                                        android=messaging.AndroidConfig(
                                            notification=messaging.AndroidNotification(
                                                sound="default",
                                                priority="high"
                                            )
                                        ),
                                        data={
                                            "category": kebutuhan,
                                            "gesture": gestur
                                        },
                                        token=tk
                                    ) for tk in fcm_tokens
                                ]
                                try:
                                    # Kirim sekaligus menggunakan send_each (Firebase Admin SDK 6.2.0+)
                                    response = messaging.send_each(messages)
                                    print(f"[FCM] {response.success_count} notifikasi terkirim dari {len(fcm_tokens)} perangkat.")
                                except Exception as e_msg:
                                    print(f"[ERROR] Gagal mengirim FCM: {e_msg}")
                            else:
                                print("[WARNING] Tidak ada FCM Token terdaftar di database.")
                        
                        last_signal_time[kebutuhan] = current_time
                    except Exception as e:
                        print(f"[ERROR] Gagal mengirim data ke Firebase: {e}")
                else:
                    last_signal_time[kebutuhan] = current_time


        firebase_queue.task_done()


def listen_config_changes() -> None:
    """Mengatur real-time listener ke konfigurasi gesture Firestore."""
    if not db:
        return

    doc_ref = db.collection("users").document(USER_ID).collection("config").document("gestures")

    def on_snapshot(doc_snapshot, changes, read_time):
        global gesture_mapping
        for doc in doc_snapshot:
            if doc.exists:
                config_data = doc.to_dict()
                # Ekstrak field mapping jika dokumen terstruktur sebagai { "mapping": {...}, "settings": {...} }
                # atau langsung gunakan data dokumen jika strukturnya langsung mapping.
                mapping_data = config_data.get("mapping", config_data)
                
                new_mapping = {}
                for kebutuhan, daftar_gestur in mapping_data.items():
                    if isinstance(daftar_gestur, list):
                        for gestur in daftar_gestur:
                            new_mapping[gestur] = kebutuhan
                
                # Update aman (in-place)
                gesture_mapping.clear()
                gesture_mapping.update(new_mapping)
                print(f"[SYNC] Real-time gesture mapping diupdate: {gesture_mapping}")

    # Pasang listener dan simpan referensinya
    global _config_watch
    _config_watch = doc_ref.on_snapshot(on_snapshot)
    print("[FIREBASE] Listener real-time untuk konfigurasi aktif.")


def start_firebase_worker() -> None:
    """Menjalankan Firebase worker di background thread daemon."""
    threading.Thread(target=firebase_worker, daemon=True, name="firebase-worker").start()
    print("[FIREBASE] Worker thread dimulai.")
    
    # Jalankan listener untuk sync otomatis dari cloud
    listen_config_changes()


def stop_firebase_worker() -> None:
    """Menghentikan Firebase worker dan listener agar proses bisa exit bersih."""
    global _config_watch
    # Hentikan worker thread dengan sentinel
    firebase_queue.put(None)
    # Unsubscribe Firestore listener
    if _config_watch is not None:
        try:
            _config_watch.unsubscribe()
            print("[FIREBASE] Listener real-time di-unsubscribe.")
        except Exception as e:
            print(f"[FIREBASE] Gagal unsubscribe listener: {e}")
        _config_watch = None
