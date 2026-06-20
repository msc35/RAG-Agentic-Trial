# Build and run:
#   docker build -t rag-agent .
#   docker run -p 8000:8000 \
#     -e OPENAI_API_KEY=sk-... \
#     -v $(pwd)/data:/app/data \
#     rag-agent
#
# data/pdfs/ and data/store/ are mounted as a volume so:
#   • PDFs live outside the image (re-ingest without rebuilding the image)
#   • The vector store persists across container restarts

FROM python:3.11-slim

# System deps needed to compile some native extensions (chromadb, tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Disable TF import (Keras 3 / tf-keras conflict, not needed for our stack)
ENV TRANSFORMERS_NO_TF=1
ENV USE_TF=0

# Copy application source
COPY src/ src/
COPY evals/ evals/

# Placeholder directories — real data is mounted via -v
RUN mkdir -p data/pdfs data/store

EXPOSE 8000

# Run as non-root for safety
RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
