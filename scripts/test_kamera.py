"""
╔══════════════════════════════════════════════════════════════╗
║   StrokeMonitor — Standalone Camera Test (Tanpa Web/UI)      ║
║   Jalankan dari root folder struk-monitor:                   ║
║       python scripts/test_kamera.py                          ║
║                                                              ║
║   Tekan Q untuk keluar.                                      ║
╚══════════════════════════════════════════════════════════════╝

Optimasi responsivitas v2:
  - pred_history maxlen: 3  (butuh 2 suara untuk konfirmasi)
  - Instant override: conf > 0.93 → langsung ganti label tanpa voting
  - Temporal stride: setiap 2 frame (bukan 3)
  - Delta movement threshold: 0.08 (lebih sensitif)
  - Partial buffer: inferensi mulai saat buffer >= 20 frame
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import threading
import queue
from collections import deque

# ══════════════════════════════════════════════════════════════════
#  KONFIGURASI
# ══════════════════════════════════════════════════════════════════
MODEL_PATH   = os.path.join(os.path.dirname(__file__), '../models/model.tflite')
LABEL_PATH   = os.path.join(os.path.dirname(__file__), '../models/labels.npy')
COOLDOWN_SEC = 5
GESTURE_MAP  = {"A": "Makan", "B": "Minum", "C": "Toilet", "D": "Tidur"}

# ── Tuning Parameter Responsivitas ────────────────────────────────
PRED_HISTORY_SIZE     = 3      # Butuh 2 suara → konfirmasi lebih cepat
TEMPORAL_STRIDE       = 2      # Inferensi tiap 2 frame (bukan 3)
DELTA_THRESHOLD       = 0.08   # Reset history saat tangan bergerak ≥ 8% lebar frame
INSTANT_CONF          = 0.93   # Di atas ini → langsung override tanpa voting
HIGH_CONF             = 0.78   # Threshold untuk masuk pred_history
LOW_CONF              = 0.38   # Di bawah ini → clear pred_history
PARTIAL_BUFFER_MIN    = 20     # Mulai inferensi dari frame ke-20 (bukan ke-30)

# ══════════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════════
print("[INIT] Memuat model TFLite...")
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
    print("[ERROR] model.tflite atau labels.npy tidak ditemukan!")
    sys.exit(1)

labels      = np.load(LABEL_PATH)
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"[OK] Model siap | Kelas: {labels}")

def predict_tflite(buf):
    """Inferensi TFLite. Padding ke 30 frame jika buf < 30."""
    arr = np.array(buf, dtype=np.float32)   # (N, 63)
    if arr.shape[0] < 30:
        # Pad dengan frame terakhir (repeat padding) agar shape (30, 63)
        pad = np.tile(arr[-1], (30 - arr.shape[0], 1))
        arr = np.vstack([arr, pad])
    data = arr[np.newaxis, ...]             # (1, 30, 63)
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# ══════════════════════════════════════════════════════════════════
#  STATE BERSAMA
# ══════════════════════════════════════════════════════════════════
_buffer       = deque(maxlen=30)
_pred_history = deque(maxlen=PRED_HISTORY_SIZE)
_last_signal  = {}

status = {
    "label"     : "Waiting...",
    "confidence": 0.0,
    "signal"    : "-",
}

_inference_queue = queue.Queue(maxsize=3)

# ══════════════════════════════════════════════════════════════════
#  INFERENCE WORKER
# ══════════════════════════════════════════════════════════════════
def inference_worker():
    while True:
        task = _inference_queue.get()
        if task is None:
            break

        buf_copy   = task['buffer']
        reset_flag = task.get('reset', False)

        if reset_flag:
            _pred_history.clear()

        pred       = predict_tflite(buf_copy)
        idx        = int(np.argmax(pred))
        conf       = float(pred[idx])
        new_label  = labels[idx]
        cur_label  = status["label"]

        # ── INSTANT OVERRIDE ────────────────────────────────────────────
        # Jika yakin sekali (conf > INSTANT_CONF), langsung ganti label
        # tanpa menunggu mayoritas voting → latensi ~0 ms
        if conf >= INSTANT_CONF:
            _pred_history.clear()
            for _ in range(PRED_HISTORY_SIZE):      # isi semua slot
                _pred_history.append(new_label)
            final_label = new_label

        # ── VOTING NORMAL ──────────────────────────────────────────────
        elif conf >= HIGH_CONF:
            _pred_history.append(new_label)
            # Jika gestur baru muncul berbeda dari saat ini → bobot ekstra
            if new_label != cur_label:
                _pred_history.append(new_label)   # double vote untuk transisi
            final_label = max(set(_pred_history), key=list(_pred_history).count)

        elif conf <= LOW_CONF:
            _pred_history.clear()
            final_label = "Analyzing..."

        else:
            # Zona abu-abu: pertahankan label lama
            final_label = cur_label if cur_label not in ("Waiting...", "Analyzing...") else "Analyzing..."

        # ── UPDATE STATUS ──────────────────────────────────────────────
        status["label"]      = str(final_label)
        status["confidence"] = conf

        # Trigger sinyal jika label valid dan cooldown terpenuhi
        now = time.time()
        if final_label in GESTURE_MAP:
            kebutuhan = GESTURE_MAP[final_label]
            if kebutuhan not in _last_signal or (now - _last_signal[kebutuhan]) > COOLDOWN_SEC:
                _last_signal[kebutuhan] = now
                status["signal"] = f"SINYAL >> {kebutuhan} (Gestur: {final_label})"
                print(f"\n  >> {status['signal']}\n")

        _inference_queue.task_done()

threading.Thread(target=inference_worker, daemon=True, name="inference-worker").start()

# ══════════════════════════════════════════════════════════════════
#  KAMERA
# ══════════════════════════════════════════════════════════════════
print("[INIT] Membuka kamera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

if not cap.isOpened():
    print("[ERROR] Kamera tidak bisa dibuka!")
    sys.exit(1)

time.sleep(1.0)
print("[OK] Kamera siap. Tekan Q untuk keluar.\n")

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.65,   # Sedikit diturunkan agar deteksi lebih responsif
    min_tracking_confidence=0.5,
)

frame_count    = 0
last_wrist_pos = None
fps_counter    = 0
fps_start      = time.time()
fps_display    = 0.0

# ══════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    fps_counter += 1

    # Hitung FPS tiap detik
    elapsed = time.time() - fps_start
    if elapsed >= 1.0:
        fps_display = fps_counter / elapsed
        fps_counter = 0
        fps_start   = time.time()

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_lms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        pts   = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
        wrist = pts[0]

        # ── DELTA MOVEMENT DETECTION (Threshold lebih sensitif: 0.08) ──
        current_wrist = np.array([wrist[0], wrist[1]])
        reset_history = False
        if last_wrist_pos is not None:
            dist = np.linalg.norm(current_wrist - last_wrist_pos)
            if dist > DELTA_THRESHOLD:
                reset_history = True
                _pred_history.clear()   # Reset langsung di main thread juga
        last_wrist_pos = current_wrist

        # ── NORMALISASI ────────────────────────────────────────────────
        centered  = pts - wrist
        hand_size = np.linalg.norm(centered[0] - centered[9]) or 1.0
        _buffer.append((centered / hand_size).flatten())

        # ── TEMPORAL STRIDING (setiap 2 frame, partial buffer dari 20) ─
        buf_len = len(_buffer)
        if buf_len >= PARTIAL_BUFFER_MIN and frame_count % TEMPORAL_STRIDE == 0:
            if not _inference_queue.full():
                _inference_queue.put_nowait({
                    'buffer': list(_buffer),
                    'reset' : reset_history,
                })
    else:
        last_wrist_pos = None
        # Jika tangan hilang lebih dari 0.5 detik, clear buffer
        if frame_count % 15 == 0:
            _buffer.clear()
            status["label"] = "Waiting..."

    # ══════════════════════════════════════════════════════════════
    #  HUD OVERLAY
    # ══════════════════════════════════════════════════════════════
    label = status["label"]
    conf  = status["confidence"]
    sig   = status["signal"]

    # Panel hitam semi-transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (640, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Warna adaptif berdasarkan confidence
    if conf >= INSTANT_CONF:
        color = (0, 255, 0)       # Hijau terang → instant confirm
    elif conf >= HIGH_CONF:
        color = (0, 255, 200)     # Cyan → voting normal
    elif conf >= 0.40:
        color = (0, 200, 255)     # Kuning → rendah
    else:
        color = (80, 80, 80)      # Abu → tidak terdeteksi

    # Baris 1: Gesture + Confidence + indikator instant
    instant_mark = " [INSTANT]" if conf >= INSTANT_CONF else ""
    cv2.putText(frame,
                f"Gesture: {label}  ({conf*100:.1f}%){instant_mark}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    # Baris 2: FPS + Buffer + Sinyal
    buf_info = f"Buf:{len(_buffer)}/30"
    cv2.putText(frame,
                f"FPS:{fps_display:.0f}  {buf_info}  |  {sig}",
                (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("StrokeMonitor — Test Kamera (Q: keluar)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n[EXIT] Menutup sistem...")
        break

# ══════════════════════════════════════════════════════════════════
#  CLEANUP
# ══════════════════════════════════════════════════════════════════
_inference_queue.put(None)
hands.close()
cap.release()
cv2.destroyAllWindows()
print("[DONE] Sistem dimatikan.")
