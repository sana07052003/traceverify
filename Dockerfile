# Works on Hugging Face Spaces (SDK: Docker, app_port: 7860), Render, Railway, Fly.io, any VM.
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
ENV HF_HOME=/tmp/hf TV_ARTIFACT_DIR=/tmp/artifacts PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
