import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

from app.services.inference import predict
from app.services.camera import VideoStream
from app.core.firebase import firebase_queue, gesture_mapping

# Inisialisasi MediaPipe versi Legacy
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Global variables
_model = None
_input_details = None
_output_details = None
_labels = None
_vs = None
_hands = None

buffer = deque(maxlen=30)
pred_history = deque(maxlen=12)
global_status = {
    "current_label": "Waiting...",
    "confidence": 0.0
}

def init_gesture_service(model, input_details, output_details, labels):
    """Inisialisasi service gesture dengan model TFLite."""
    global _model, _input_details, _output_details, _labels, _vs, _hands
    _model = model
    _input_details = input_details
    _output_details = output_details
    _labels = labels

    # MediaPipe Legacy Hands - Set True karena request dari browser per-frame
    _hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    print("[MEDIAPIPE] Legacy Hands siap.")

    print("[CAMERA] Menginisialisasi kamera...")
    _vs = VideoStream(src=0).start()
    print("[CAMERA] Kamera siap.")


def generate_frames():
    """Generator untuk streaming video ke FastAPI."""
    global global_status
    while True:
        frame = _vs.read()
        if frame is None:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Proses dengan MediaPipe Legacy
        results = _hands.process(rgb)

        label, conf = _process_legacy_detection(frame, results)

        # ── Update global status ───────────────────────────────────────────────
        global_status["current_label"] = str(label)
        global_status["confidence"] = conf

        # ── Annotate frame ─────────────────────────────────────────────────────
        cv2.putText(frame, f"{label} ({conf:.2f})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ── Encode ke JPEG ─────────────────────────────────────────────────────
        ret, jpeg = cv2.imencode(".jpg", frame)
        if ret:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
            )


def process_external_frame(frame: np.ndarray) -> dict:
    """Memproses frame dari browser menggunakan logika legacy."""
    global _hands
    
    # Flip di sini agar sesuai dengan training data model
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _hands.process(rgb)
    
    label, conf = _process_legacy_detection(frame, results)
    
    # Encode frame yang sudah digambar ke Base64
    _, buffer_img = cv2.imencode('.jpg', frame)
    import base64
    img_base64 = base64.b64encode(buffer_img).decode('utf-8')
    
    return {
        "label": label,
        "confidence": float(conf),
        "annotated_image": img_base64,
        "firebase_sync": "active"
    }


def _process_legacy_detection(frame: np.ndarray, results) -> tuple:
    """Logika inti klasifikasi menggunakan hasil MediaPipe Legacy."""
    label = "Waiting..."
    conf = 0.0

    if results.multi_hand_landmarks:
        hand_lms = results.multi_hand_landmarks[0]
        
        print(f"[DEBUG] Tangan terdeteksi! Buffer: {len(buffer)+1}/30")

        # Gambar skeleton (opsional di backend karena UI juga menggambar)
        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        # Preprocessing sekuensial
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
        wrist = pts[0]
        centered = pts - wrist
        hand_size = np.linalg.norm(centered[0] - centered[9])
        if hand_size == 0: hand_size = 1.0
        final_norm = (centered / hand_size).flatten()
        
        buffer.append(final_norm)

        if len(buffer) == 30:
            pred = predict(_model, _input_details, _output_details, list(buffer))
            idx = int(np.argmax(pred))
            conf = float(pred[idx])
            print(f"[AI] Raw Prediction: {_labels[idx]} ({conf:.2f})")

            if conf > 0.75:
                pred_history.append(_labels[idx])
                if conf > 0.95: pred_history.append(_labels[idx])
            elif conf < 0.40:
                pred_history.clear()

            if len(pred_history) > 0:
                label = max(set(pred_history), key=list(pred_history).count)
                if label in gesture_mapping:
                    firebase_queue.put({
                        "action": "push_history",
                        "kebutuhan": gesture_mapping[label],
                        "gestur": label,
                    })
            else:
                label = "Analyzing..."
    
    return label, conf
