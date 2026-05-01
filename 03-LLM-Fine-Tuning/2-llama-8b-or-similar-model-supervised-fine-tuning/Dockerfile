FROM python:3.10-slim

RUN pip install --no-cache-dir fastapi uvicorn transformers torch

COPY . /app
WORKDIR /app

EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
