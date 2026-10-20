# Pinned to a Debian release: the unpinned tag moved to trixie, which dropped libgl1-mesa-glx
FROM python:3.10-slim-bookworm

WORKDIR /app

# Install system dependencies for OpenCV and PyTorch (libgl1 replaces libgl1-mesa-glx)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
