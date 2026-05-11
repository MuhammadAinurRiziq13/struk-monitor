# Dokumentasi Arsitektur StrokeMonitor AI

StrokeMonitor adalah sistem pemantauan pasien stroke berbasis AI yang mendeteksi bahasa isyarat (gestur tangan) secara real-time dan mengirimkan notifikasi instan ke perawat/caregiver melalui aplikasi mobile.

---

## 1. Arsitektur Sistem

Sistem ini terdiri dari tiga komponen utama:
1.  **AI Backend (Python/FastAPI)**: Berjalan di perangkat lokal (seperti Laptop atau Raspberry Pi) dengan kamera.
2.  **Firebase Cloud**: Sebagai jembatan komunikasi data (Firestore) dan pengiriman notifikasi (FCM).
3.  **Mobile App (Android/Kotlin)**: Aplikasi yang dibawa oleh perawat untuk menerima peringatan.

### Diagram Alur Kerja
`Kamera` → `AI Backend` → `Firestore/FCM` → `Mobile App`

---

## 2. Komponen Backend AI (Mediapipe Module)

Backend bertanggung jawab untuk memproses video stream dan mendeteksi gestur.

*   **MediaPipe**: Digunakan untuk mengekstrak 21 titik koordinat (landmarks) tangan manusia secara 3D.
*   **TFLite Model**: Model Machine Learning ringan (MLP) yang telah dilatih untuk mengklasifikasikan koordinat tangan menjadi label (seperti: Makan, Minum, Toilet, Tidur).
*   **FastAPI**: Web server yang menyediakan:
    *   `Video Feed`: Monitoring visual pasien via browser.
    *   `Status API`: Mengetahui kondisi server secara real-time.
*   **Firebase Worker**: Thread khusus yang berjalan di latar belakang untuk mengirim data ke cloud tanpa mengganggu proses deteksi AI.

---

## 3. Komunikasi & Database (Firebase)

Firebase digunakan agar sistem bisa diakses dari mana saja (tidak terbatas jaringan lokal).

*   **Cloud Firestore**: Menyimpan data riwayat notifikasi di koleksi `notifications`. Setiap dokumen berisi kategori (Makan/Minum), nama gestur, confidence score, dan status (`pending`/`handled`).
*   **FCM (Firebase Cloud Messaging)**: Mekanisme pengiriman notifikasi instan.
    *   **Trigger**: Terjadi saat AI Backend berhasil mendeteksi gestur dengan confidence tinggi.
    *   **Proses**: 
        1. AI Backend mengambil `fcm_token` unik milik HP perawat dari Firestore.
        2. Backend mengirim payload notifikasi (Title, Body, Sound) menggunakan library `firebase_admin.messaging`.
        3. Google FCM Server meneruskan pesan tersebut ke HP target secara real-time.
    *   **Keunggulan**: Notifikasi tetap muncul meskipun HP dalam keadaan terkunci atau aplikasi sedang tidak dibuka (background).
*   **Real-time Sync**: Selain notifikasi, aplikasi mobile menggunakan `SnapshotListener` untuk memantau perubahan status bantuan secara real-time langsung dari database Firestore.

---

## 4. Aplikasi Mobile (Android)

Dibuat menggunakan teknologi terbaru (Native Android dengan Jetpack Compose).

*   **Dashboard Utama**: Menampilkan sinyal terbaru dari pasien dengan animasi "detak jantung" (pulse) jika ada bantuan yang diperlukan.
*   **Swipe-to-Confirm**: Fitur keamanan untuk menandai bantuan telah ditangani. Perawat harus menggeser slider ke kanan (seperti *slide to unlock*) untuk memastikan tindakan tidak dilakukan secara tidak sengaja.
*   **Notification Service**: Menangani pesan masuk dari Firebase dan memunculkan pop-up serta suara di HP.

---

## 5. Cara Kerja Program (Step-by-Step)

1.  **Deteksi**: Kamera menangkap gerakan tangan pasien. MediaPipe mengubahnya menjadi data angka (koordinat).
2.  **Klasifikasi**: AI menentukan bahwa gestur tersebut adalah "Makan" (confidence > 80%).
3.  **Pengiriman**: Backend mengirim data ke Firestore dan memicu push notification lewat FCM.
4.  **Peringatan**: HP perawat berbunyi dan muncul notifikasi "Pasien membutuhkan bantuan: Makan".
5.  **Interaksi**: Dashboard aplikasi mobile berkedip merah/oranye.
6.  **Penanganan**: Perawat mendatangi pasien, lalu melakukan **Swipe** pada slider di aplikasi. Status di database berubah menjadi `handled` dan dashboard berhenti berkedip.

---

## 6. Daftar Gestur yang Didukung
Sistem saat ini dilatih untuk mengenali huruf isyarat yang dipetakan ke kebutuhan:
*   **Makan**: Gestur huruf 'A' / 'B'
*   **Minum**: Gestur huruf 'C'
*   **Toilet**: Gestur huruf 'D'
*   **Tidur**: Gestur huruf 'L'
*   *(Dapat dikustomisasi melalui menu konfigurasi)*
