# Base image: slim Debian with Python 3.11 preinstalled
FROM python:3.11-slim

# Hugging Face Spaces runs the container as user ID 1000, so create that user
# and work as it - otherwise the app can't write its index and model cache.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed

WORKDIR /home/user/app

# Install dependencies first. Docker caches this layer, so later builds
# skip reinstalling packages unless requirements.txt changes.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the application code and the PDF
COPY --chown=user . .

# Build the vector index (and download the embedding model) at build time,
# so the app starts ready to answer. Needs data/document.pdf in the build context.
RUN python ingest.py

# Spaces routes traffic to port 7860
EXPOSE 7860

EXPOSE 7860

# Hosting platforms like Render pass the port in $PORT; fall back to 7860 locally.
# 0.0.0.0 makes the server reachable from outside the container.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]