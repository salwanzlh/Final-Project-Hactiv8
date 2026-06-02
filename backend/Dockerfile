FROM python:3.12-slim

# ffmpeg untuk konversi WebM → PCM (dibutuhkan STT ElevenLabs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Salin uv langsung dari image resminya
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependensi dulu (layer cache — tidak rebuild kalau kode berubah)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Salin source code
COPY backend/ ./backend/
COPY main.py ./

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
