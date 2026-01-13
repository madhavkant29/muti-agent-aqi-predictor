FROM python:3.12-slim

WORKDIR /app

# install system deps (for ML libs safety)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 8501

# default command = backend (can be overridden in compose)
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]