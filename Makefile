.PHONY: app notebooks previews test compose-test

app:
	uvicorn app.main:app --reload

notebooks:
	python scripts/generate_notebooks.py

previews:
	python scripts/build_previews.py

test:
	pytest -q

compose-test:
	docker compose run --rm test

