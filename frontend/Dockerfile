FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir streamlit==1.35.0

COPY . .

HEALTHCHECK --interval=30s --timeout=5s CMD curl -sf http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
