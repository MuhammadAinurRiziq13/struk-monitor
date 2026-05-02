# Gunakan Python 3.10 slim untuk efisiensi ukuran image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Instal dependensi sistem yang dibutuhkan oleh OpenCV, MediaPipe, dan TFLite Flex Ops
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt terlebih dahulu untuk memanfaatkan caching Docker layer
COPY requirements.txt .

# Instal dependensi Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir tensorflow==2.16.1 mediapipe==0.10.9 python-dotenv fastapi uvicorn firebase-admin opencv-python-headless numpy==1.26.4

# Copy seluruh kode aplikasi
COPY . .

# Expose port FastAPI
EXPOSE 8000

# Command untuk menjalankan aplikasi
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
