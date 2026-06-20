# Convenience commands. Use a Python 3.11+ interpreter.
# Override the interpreter if needed:  make ingest PYTHON=python3.11
PYTHON ?= python3

.PHONY: help install ingest api eval test clean frontend

help:
	@echo "Targets:"
	@echo "  make install   Install dependencies from requirements.txt"
	@echo "  make ingest    Build the vector store from PDFs in data/pdfs/"
	@echo "  make api       Run the FastAPI service (Phase 5)"
	@echo "  make eval      Run the evaluation harness (Phase 6)"
	@echo "  make test      Run the pytest suite (Phase 8)"
	@echo "  make clean     Delete the persisted vector store"

install:
	$(PYTHON) -m pip install -r requirements.txt

ingest:
	$(PYTHON) -m src.ingest

# Rebuild from scratch (clears the collection first).
ingest-rebuild:
	$(PYTHON) -m src.ingest --rebuild

api:
	$(PYTHON) -m uvicorn src.api:app --reload --port 8000

eval:
	$(PYTHON) -m evals.run_eval

test:
	$(PYTHON) -m pytest -q

frontend:
	cd frontend && npm install && npm run dev

clean:
	rm -rf data/store/* && touch data/store/.gitkeep
