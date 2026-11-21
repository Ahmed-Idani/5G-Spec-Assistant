# One-command setup and run. See README.md.
PY := .venv/bin/python

.PHONY: setup data index api web

setup:            ## create .venv, install Python (CPU torch) and frontend dependencies
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
	.venv/bin/pip install -r requirements.txt
	cd web && npm install

data:             ## download the two 3GPP specs and parse them into chunks
	$(PY) -m ingest.download_specs
	$(PY) -m ingest.parse_docx

index:            ## build the vector (ChromaDB) and BM25 indexes (~30-45 min on CPU)
	$(PY) -m ingest.build_index

api:              ## start the backend on http://localhost:8000
	.venv/bin/uvicorn api.server:app --port 8000

web:              ## start the web app on http://localhost:5173
	cd web && npm run dev
