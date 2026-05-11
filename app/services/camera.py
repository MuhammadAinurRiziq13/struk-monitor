"""
Service kamera: VideoStream thread-safe dengan FPS cap.
Lock digunakan untuk mencegah race condition antara reader thread
dan consumer thread (generator / mediapipe worker).
"""
import cv2
import time
import threading
import numpy as np


class VideoStream:
    """
    Threaded, thread-safe video capture wrapper untuk OpenCV.
    - Lock melindungi akses frame dari race condition.
    - FPS cap mencegah reader thread memakan 100% CPU.
    - Warm-up otomatis setelah start().
    """

    def __init__(self, src: int = 0, width: int = 640, height: int = 480) -> None:
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.stream.isOpened():
            # Fallback jika CAP_DSHOW gagal (misal di Linux/Docker)
            self.stream = cv2.VideoCapture(src)
            
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.width = width
        self.height = height
        self.grabbed, self.frame = self.stream.read()
        
        if not self.grabbed:
            print(f"[WARN] Kamera (index {src}) tidak terdeteksi atau tidak bisa dibuka.")
            self.frame = self._create_placeholder("KAMERA TIDAK TERDETEKSI")
            
        self.stopped = False
        self._lock = threading.Lock()

    def _create_placeholder(self, text: str):
        """Membuat frame hitam dengan teks peringatan."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(img, text, (50, self.height // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, "Cek Docker Device / USBIPD", (50, self.height // 2 + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        return img

    def start(self, warmup: float = 1.0) -> "VideoStream":
        """Memulai background thread pembaca frame."""
        threading.Thread(target=self._update, daemon=True, name="camera-reader").start()
        if self.stream.isOpened():
            time.sleep(warmup)
        return self

    def _update(self) -> None:
        """Loop internal yang terus membaca frame dari kamera dengan FPS cap."""
        interval = 1.0 / 60
        while not self.stopped:
            if not self.stream.isOpened():
                time.sleep(1.0)
                continue
                
            grabbed, frame = self.stream.read()
            if not grabbed:
                # Jika kamera terputus, gunakan placeholder
                with self._lock:
                    self.frame = self._create_placeholder("KAMERA TERPUTUS")
                time.sleep(1.0)
                continue
                
            with self._lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(interval)

    def read(self):
        """Mengembalikan salinan frame terakhir secara thread-safe."""
        with self._lock:
            if self.frame is None:
                return self._create_placeholder("INITIALIZING...")
            return self.frame.copy()

    def stop(self) -> None:
        """Menghentikan thread pembaca dan melepas resource kamera."""
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()
