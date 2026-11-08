.PHONY: install test cov lint run docker-build docker-up clean

# Windows Git Bash 的 venv 布局；Linux/macOS 把 PY 改成 .venv/bin/python
PY := .venv/Scripts/python.exe
PIP_MIRROR := https://mirrors.aliyun.com/pypi/simple/

install:
	python -m venv .venv
	$(PY) -m pip install -i $(PIP_MIRROR) -r requirements.txt
	$(PY) -m pip install -i $(PIP_MIRROR) -r requirements-rerank.txt

test:
	$(PY) -m pytest

cov:
	$(PY) -m pytest --cov=app --cov-report=term-missing

lint:
	$(PY) -m ruff check app tests

run:
	$(PY) -m uvicorn app.main:app --port 8000 --reload

docker-build:
	docker compose build

docker-up:
	docker compose up -d

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} \; 2>/dev/null || true
