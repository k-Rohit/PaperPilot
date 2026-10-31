.PHONY: help start stop restart status logs health setup format lint test test-cov clean dev dev-stop api

start:
	docker compose up --build -d

stop:
	docker compose down

status:
	docker compose ps

logs:
	docker compose logs -f

health:
	curl http://localhost:8000/api/v1/health

# Dev mode: only the infra the API needs locally (Redis; Postgres is Supabase and
# Pinecone is hosted), run the API itself with `make api` in a separate terminal —
# fast reload, no Docker rebuild per code change.
dev:
	docker compose up -d redis

dev-stop:
	docker compose stop redis

api:
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
