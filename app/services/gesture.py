"""
Service gesture: Full-async AI pipeline.

Arsitektur thread:
  camera-reader  →  [frame_queue]  →  mediapipe-worker  →  [inference_queue]
                                                         →  inference-worker  →  global_status
  generate_frames() hanya membaca annotated_frame (sudah digambar skeleton) dan encode JPEG.
  Tidak ada blocking call di dalam generator → stream MJPEG selalu lancar.
"""
import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import queue
from collections import deque

from app.services.inference import predict
from app.services.camera import VideoStream
from app.core.firebase import firebase_queue, gesture_mapping, global_status

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# ── Tuning Parameter Responsivitas ───────────────────────────────────────────
PRED_HISTORY_SIZE  = 3      # Butuh 2 suara → konfirmasi lebih cepat
TEMPORAL_STRIDE    = 2      # Inferensi tiap 2 frame (bukan 3)
DELTA_THRESHOLD    = 0.08   # Reset history saat tangan bergerak >= 8% lebar frame
INSTANT_CONF       = 0.93   # Di atas ini → instant override tanpa voting
HIGH_CONF          = 0.78   # Threshold masuk pred_history
LOW_CONF           = 0.38   # Di bawah ini → hapus pred_history
PARTIAL_BUFFER_MIN = 20     # Mulai inferensi dari frame ke-20 (bukan ke-30)


# ── State module-level ────────────────────────────────────────────────────────
_model          = None
_input_details  = None
_output_details = None
_labels         = None
_vs             = None

# Buffer sekuensial dan histori prediksi
_buffer      = deque(maxlen=30)
_pred_history = deque(maxlen=PRED_HISTORY_SIZE)

# Queue antar thread (ukuran kecil → drop otomatis jika penuh, tidak blocking)
_frame_queue     = queue.Queue(maxsize=2)  # raw frame → mediapipe worker
_inference_queue = queue.Queue(maxsize=2)  # keypoint buffer → inference worker
_annotated_queue = queue.Queue(maxsize=2)  # annotated frame → generator MJPEG

# Lock untuk akses buffer sekuensial
_buffer_lock = threading.Lock()

# ── Worker 1: MediaPipe ───────────────────────────────────────────────────────
def _mediapipe_worker():
    """Memproses frame dengan MediaPipe di thread terpisah."""
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    frame_count    = 0
    last_wrist_pos = None

    while True:
        try:
            frame = _frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if frame is None:
            break

        frame_count += 1
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]

            # Gambar skeleton pada frame
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            # Preprocessing keypoint
            pts   = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
            wrist = pts[0]

            # ── DELTA MOVEMENT (threshold lebih sensitif: 0.08) ───────────────
            current_wrist = np.array([wrist[0], wrist[1]])
            reset_history = False
            if last_wrist_pos is not None:
                dist = np.linalg.norm(current_wrist - last_wrist_pos)
                if dist > DELTA_THRESHOLD:
                    reset_history = True
                    _pred_history.clear()  # Reset langsung di mediapipe thread
            last_wrist_pos = current_wrist

            # Normalisasi
            centered  = pts - wrist
            hand_size = np.linalg.norm(centered[0] - centered[9])
            if hand_size == 0:
                hand_size = 1.0
            final_norm = (centered / hand_size).flatten()

            with _buffer_lock:
                _buffer.append(final_norm)
                buffer_len = len(_buffer)

            # Temporal Striding: kirim ke inference setiap 2 frame, mulai dari frame 20
            if buffer_len >= PARTIAL_BUFFER_MIN and frame_count % TEMPORAL_STRIDE == 0:
                if not _inference_queue.full():
                    with _buffer_lock:
                        buf_copy = list(_buffer)
                    _inference_queue.put_nowait({'buffer': buf_copy, 'reset': reset_history})
        else:
            last_wrist_pos = None  # Reset jika tangan hilang

        # Annotate teks status pada frame
        label = global_status.get("current_label", "Waiting...")
        conf  = global_status.get("confidence", 0.0)
        cv2.putText(
            frame, f"{label} ({conf:.2f})",
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        )

        # Kirim annotated frame ke queue generator
        if not _annotated_queue.full():
            _annotated_queue.put_nowait(frame)

        _frame_queue.task_done()

    hands.close()


# ── Worker 2: Inference ───────────────────────────────────────────────────────
def _inference_worker():
    """Melakukan prediksi model TFLite di thread terpisah."""
    while True:
        try:
            task = _inference_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if task is None:
            break

        buf_copy   = task['buffer']
        reset_flag = task.get('reset', False)

        if reset_flag:
            _pred_history.clear()

        # Padding jika buffer belum penuh 30 frame
        arr = np.array(buf_copy, dtype=np.float32)
        if arr.shape[0] < 30:
            pad = np.tile(arr[-1], (30 - arr.shape[0], 1))
            arr = np.vstack([arr, pad])

        pred      = predict(_model, _input_details, _output_details, list(arr))
        idx       = int(np.argmax(pred))
        conf      = float(pred[idx])
        new_label = _labels[idx]
        cur_label = global_status.get("current_label", "Waiting...")

        # ── INSTANT OVERRIDE ───────────────────────────────────────────────────
        if conf >= INSTANT_CONF:
            _pred_history.clear()
            for _ in range(PRED_HISTORY_SIZE):
                _pred_history.append(new_label)
            final_label = new_label

        # ── VOTING NORMAL ──────────────────────────────────────────────────────
        elif conf >= HIGH_CONF:
            _pred_history.append(new_label)
            if new_label != cur_label:        # Double vote untuk transisi gestur
                _pred_history.append(new_label)
            final_label = max(set(_pred_history), key=list(_pred_history).count)

        elif conf <= LOW_CONF:
            _pred_history.clear()
            final_label = "Analyzing..."

        else:
            final_label = cur_label if cur_label not in ("Waiting...", "Analyzing...") else "Analyzing..."

        if len(_pred_history) > 0 and final_label in gesture_mapping:
            firebase_queue.put_nowait({
                'action'   : 'push_history',
                'kebutuhan': gesture_mapping[final_label],
                'gestur'   : final_label,
                'confidence': conf,
            })

        global_status["current_label"] = str(final_label)
        global_status["confidence"]    = conf
        _inference_queue.task_done()


# ── Camera feeder ─────────────────────────────────────────────────────────────
def _camera_feeder():
    """Membaca frame dari VideoStream dan memasukkannya ke frame_queue."""
    while True:
        if _vs is None or _vs.stopped:
            time.sleep(0.05)
            continue
        frame = _vs.read()
        if frame is None:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1)
        if not _frame_queue.full():
            _frame_queue.put_nowait(frame)
        else:
            # Drop frame lama jika queue penuh (ambil yang baru)
            try:
                _frame_queue.get_nowait()
            except queue.Empty:
                pass
            _frame_queue.put_nowait(frame)


# ── Init ──────────────────────────────────────────────────────────────────────
def init_gesture_service(model, input_details, output_details, labels):
    """Inisialisasi seluruh pipeline AI secara asinkron."""
    global _model, _input_details, _output_details, _labels, _vs

    _model          = model
    _input_details  = input_details
    _output_details = output_details
    _labels         = labels

    print("[CAMERA] Menginisialisasi kamera...")
    _vs = VideoStream(src=0).start(warmup=1.5)
    print("[CAMERA] Kamera siap.")

    # Start semua worker thread
    threading.Thread(target=_camera_feeder,   daemon=True, name="camera-feeder").start()
    threading.Thread(target=_mediapipe_worker, daemon=True, name="mediapipe-worker").start()
    threading.Thread(target=_inference_worker, daemon=True, name="inference-worker").start()
    print("[AI] Semua worker thread aktif.")


def shutdown_gesture_service():
    """Membersihkan resource saat aplikasi dimatikan."""
    global _vs
    if _vs is not None:
        print("[AI] Menghentikan kamera...")
        _vs.stop()


import asyncio

# ── Generator untuk MJPEG stream ──────────────────────────────────────────────
async def generate_frames():
    """
    Generator MJPEG: hanya encode frame yang sudah di-annotate oleh mediapipe-worker.
    Zero blocking call → stream selalu responsif.
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75]  # 75% sudah cukup untuk web stream

    while True:
        try:
            frame = _annotated_queue.get_nowait()
        except queue.Empty:
            # Jika tidak ada frame, tidur sebentar (non-blocking)
            await asyncio.sleep(0.03)
            continue

        ret, jpeg = cv2.imencode(".jpg", frame, encode_params)
        if ret:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
            )
        # Beri kesempatan event loop memproses sinyal shutdown
        await asyncio.sleep(0.001)


# ── Fungsi helper untuk endpoint /predict (jika masih digunakan) ──────────────
def process_external_frame(frame: np.ndarray) -> dict:
    """Proses frame sinkron (fallback untuk POST /predict)."""
    hands = mp_hands.Hands(
        static_image_mode=True, max_num_hands=1,
        min_detection_confidence=0.7,
    )
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    hands.close()

    label, conf = "Waiting...", 0.0
    if results.multi_hand_landmarks:
        hand_lms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
        pts       = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
        wrist     = pts[0]
        centered  = pts - wrist
        hand_size = np.linalg.norm(centered[0] - centered[9]) or 1.0
        final_norm = (centered / hand_size).flatten()

        with _buffer_lock:
            _buffer.append(final_norm)
            buf_copy = list(_buffer) if len(_buffer) == 30 else None

        if buf_copy:
            pred  = predict(_model, _input_details, _output_details, buf_copy)
            idx   = int(np.argmax(pred))
            conf  = float(pred[idx])
            if conf > 0.75:
                _pred_history.append(_labels[idx])
            elif conf < 0.40:
                _pred_history.clear()
            if _pred_history:
                label = max(set(_pred_history), key=list(_pred_history).count)
            else:
                label = "Analyzing..."

    import base64
    _, buf_img = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return {
        "label"          : label,
        "confidence"     : float(conf),
        "annotated_image": base64.b64encode(buf_img).decode('utf-8'),
        "firebase_sync"  : "active",
    }
