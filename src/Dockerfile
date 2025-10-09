FROM python:3.12.slim
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get upgrade -y && pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]