.PHONY: install dev test lint run clean

install:
	uv sync

install-dev:
	uv sync --group dev

dev:
	uv run python -m src.cli analyze --scope full

run:
	uv run python -m src.cli analyze --scope full --data data/

chat:
	uv run python -m src.cli chat

test:
	GPA_DEMO_MODE=true GPA_LLM_API_KEY=sk-test-placeholder GPA_DB_URL=sqlite+aiosqlite:///./test.db \
	uv run python -m pytest tests/ -v --tb=short

test-fast:
	GPA_DEMO_MODE=true GPA_LLM_API_KEY=sk-test-placeholder GPA_DB_URL=sqlite+aiosqlite:///./test.db \
	uv run python -m pytest tests/ -x -q

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

lint-fix:
	uv run ruff check src/ tests/ --fix
	uv run ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf reports/*.md reports/*.png

setup-data:
	uv run python -c "from src.tools.common.data_loader import DataLoader; dl = DataLoader(); [dl.load_sample_data(n) for n in ['user_behavior','funnel','subsidy_experiment','retention','ad_campaign','touchpoint_journey','seasonal_history']]"

docker-build:
	docker build -f deploy/Dockerfile -t growth-pilot-api .

docker-run:
	docker run --rm -it -p 8000:8000 --env-file .env -v $(PWD)/data:/app/data growth-pilot-api

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f web

web:
	uv run uvicorn src.web:app --host 0.0.0.0 --port 8000 --reload
