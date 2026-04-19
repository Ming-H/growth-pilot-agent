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
	uv run pytest tests/unit/ -v --tb=short

test-all:
	uv run pytest tests/ -v --tb=short

lint:
	uv run ruff check src/

lint-fix:
	uv run ruff check src/ --fix

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf reports/*.md reports/*.png

setup-data:
	uv run python -c "from src.tools.common.data_loader import DataLoader; dl = DataLoader(); [dl.load_sample_data(n) for n in ['user_behavior','funnel','subsidy_experiment','retention','ad_campaign','touchpoint_journey','seasonal_history']]"

docker-build:
	docker build -f deploy/Dockerfile -t growth-pilot-agent .

docker-run:
	docker run --rm -it -v $(PWD)/data:/app/data growth-pilot-agent
