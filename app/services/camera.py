"""
Service kamera: VideoStream dengan threading agar frame capture tidak memblokir
loop AI. Setiap frame dibaca di background thread, loop utama hanya mengambil
frame terakhir yang sudah siap.
"""
import cv2
import threading


class VideoStream:
    """
    Threaded video capture wrapper untuk OpenCV.
    Menghindari bottleneck I/O kamera pada loop inferensi utama.
    """

    def __init__(self, src: int = 0, width: int = 640, height: int = 480) -> None:
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self) -> "VideoStream":
        """Memulai background thread pembaca frame."""
        threading.Thread(target=self._update, daemon=True, name="camera-reader").start()
        return self

    def _update(self) -> None:
        """Loop internal yang terus membaca frame dari kamera."""
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed, self.frame = self.stream.read()

    def read(self):
        """Mengembalikan frame terakhir yang berhasil dibaca."""
        return self.frame

    def stop(self) -> None:
        """Menghentikan thread pembaca dan melepas resource kamera."""
        self.stopped = True
        self.stream.release()
