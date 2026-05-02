# StrokeMonitor — Backend AI (FastAPI Modular)

Backend AI untuk sistem monitoring pasien stroke berbasis **deteksi bahasa isyarat (sign language)**. Project ini telah dikembangkan dengan struktur modular menggunakan **FastAPI**, **MediaPipe Tasks API**, dan **TensorFlow Lite**.

HALOOOOO


## 🗂️ Struktur Project

```text
Mediapipe/
├── app/
│   ├── api/
│   │   └── routes/         # Endpoint API (Status, Video Feed, Sync)
│   ├── core/
│   │   ├── config.py       # Konfigurasi via .env
│   │   └── firebase.py     # Integrasi Firebase & Worker Thread
│   ├── services/
│   │   ├── camera.py       # Threaded Camera Stream
│   │   ├── inference.py    # TFLite Inference Logic
│   │   └── gesture.py      # MediaPipe & Gesture Logic
│   └── main.py             # Entry Point Aplikasi
├── models/
│   ├── model.tflite        # Model Klasifikasi Gestur
│   ├── labels.npy          # Label Kelas Gestur
│   └── hand_landmarker.task # Model MediaPipe Hand Landmarker
├── scripts/
│   └── convert_tflite.py   # Script Konversi Model (opsional)
├── .env                    # Konfigurasi Lokal (USER_ID, dll)
├── requirements.txt
└── README.md
```

---

---

## 🚀 Cara Menjalankan

### Opsi A: Menggunakan Docker (Direkomendasikan)
Opsi ini paling stabil karena semua dependensi sudah terkonfigurasi di dalam container.

1.  **Build Image**:
    ```bash
    docker compose build
    ```
2.  **Jalankan Container**:
    ```bash
    docker compose up
    ```

### Opsi B: Jalankan Lokal (Manual)
1.  **Persiapan Environment**:
    Pastikan virtual environment aktif dan dependencies terinstal:
    ```bash
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```
2.  **Konfigurasi .env**:
    Salin `.env.example` menjadi `.env` dan sesuaikan nilainya.
3.  **Jalankan Server**:
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```

---

## 📡 API Endpoints

| Endpoint | Method | Deskripsi |
| :--- | :--- | :--- |
| `/` | GET | Health check server |
| `/status` | GET | Status gestur terkini (JSON) |
| `/video_feed` | GET | Live stream video dengan overlay AI |
| `/sync` | POST | Paksa sinkronisasi config dari Firebase |

---

## 🤟 Gestur yang Didukung
Sistem mendeteksi 10 kelas gestur: **A, B, C, D, F, I, L, V, W, Y**.
Mapping default:
*   A → Makan
*   B → Minum
*   C → Toilet
*   D → Tidur

Mapping dapat diubah secara dinamis melalui Firebase Firestore.

---

## ⚠️ Persyaratan Sistem
*   **MediaPipe Hand Landmarker**: Membutuhkan file `models/hand_landmarker.task`.
*   **Firebase**: Membutuhkan file `config/firebase_credentials.json`. Jika tidak ada, sistem akan berjalan dalam mode **OFFLINE**.
