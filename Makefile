.PHONY: install ingest validate features train test api docker-up docker-down clean

install:
	pip install -r requirements.txt -r requirements-dev.txt

ingest:
	python -m src.ingest --use-sample

validate:
	python -m src.validate_data

features:
	python -m src.features

train:
	python -m src.train --no-register

test:
	pytest -q

api:
	python -m app.app

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ src/__pycache__ app/__pycache__ tests/__pycache__
