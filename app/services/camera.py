"""
Service kamera: VideoStream thread-safe dengan FPS cap.
Lock digunakan untuk mencegah race condition antara reader thread
dan consumer thread (generator / mediapipe worker).
"""
import cv2
import time
import threading


class VideoStream:
    """
    Threaded, thread-safe video capture wrapper untuk OpenCV.
    - Lock melindungi akses frame dari race condition.
    - FPS cap mencegah reader thread memakan 100% CPU.
    - Warm-up otomatis setelah start().
    """

    def __init__(self, src: int = 0, width: int = 640, height: int = 480) -> None:
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)  # CAP_DSHOW lebih cepat di Windows
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, 30)               # Minta 30 FPS dari driver
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)         # Buffer minimal → selalu frame terbaru
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self._lock = threading.Lock()

    def start(self, warmup: float = 1.0) -> "VideoStream":
        """Memulai background thread pembaca frame."""
        threading.Thread(target=self._update, daemon=True, name="camera-reader").start()
        time.sleep(warmup)  # Beri waktu kamera warm-up
        return self

    def _update(self) -> None:
        """Loop internal yang terus membaca frame dari kamera dengan FPS cap."""
        interval = 1.0 / 60  # Baca maks 60x/detik, cukup untuk sumber 30fps
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stop()
                break
            with self._lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(interval)

    def read(self):
        """Mengembalikan salinan frame terakhir secara thread-safe."""
        with self._lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self) -> None:
        """Menghentikan thread pembaca dan melepas resource kamera."""
        self.stopped = True
        self.stream.release()
