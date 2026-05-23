FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY src ./src
COPY data/raw ./data/raw

RUN mkdir -p data/metadata data/processed model mlruns

RUN python -m src.ingest --use-bundled && \
    python -m src.validate_data && \
    python -m src.features && \
    python -m src.train --no-register

EXPOSE 5000

CMD ["python", "-m", "app.app"]