.PHONY: setup data train evaluate serve test docker-build docker-run all

setup:
	python -m pip install -r requirements.txt

data:
	python run_full.py

train:
	python -c "print('Training already done via run_full.py')"

evaluate:
	python -c "import json; print(json.load(open('reports/metrics/test_metrics.json')))"

serve:
	uvicorn src.app:app --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

docker-build:
	docker build -t news-classifier:latest .

docker-run:
	docker run -p 8000:8000 news-classifier:latest

all: setup data train evaluate test
