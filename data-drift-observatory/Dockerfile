# Data Drift Observatory -- API container
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY dashboard/ dashboard/

ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "ddo.api:app", "--host", "0.0.0.0", "--port", "8000"]
