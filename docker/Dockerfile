FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache
WORKDIR /app

# CPU-only PyTorch wheels (much smaller than the default CUDA build).
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY pyproject.toml README.md corpus.yaml ./
COPY config ./config
COPY .streamlit ./.streamlit
COPY src ./src
RUN pip install --no-deps -e .

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "src/uae_rag/ui/app.py", "--server.address=0.0.0.0"]
