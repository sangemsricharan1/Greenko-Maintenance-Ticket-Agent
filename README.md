# Greenko – AI Agent for Maintenance Ticket Prioritization

A runnable decision-support prototype for classifying renewable-energy maintenance tickets, assigning severity, suggesting a probable root cause and next action, and routing the ticket to a responsible team.

## What is implemented

- FastAPI backend with `/api/health`, `/api/demo`, `/api/tickets`, `/api/run`, and `/api/evaluate`.
- Browser UI served from the same FastAPI process. Includes sample tickets, custom ticket input, decision output, agent trace and evaluation view.
- NLP-assisted category suggestion: keyword rules first, then TF-IDF (unigram + bigram) cosine similarity against the supplied examples; low-evidence inputs fall back to `general_maintenance`.
- Explainable severity rules with Critical / High / Medium / Low levels.
- Root-cause hypotheses, recommended actions, priority queues and team routing.
- Sequential tool-style orchestration: `classify_ticket → score_severity → infer_root_cause → recommend_action → route_ticket`.
- Leave-one-out category retrieval evaluation script exposed through the API.
- Automated API/engine tests.

## Important scope and honesty note

The supplied CSV contains **six labelled examples**. Each category occurs only once. This is not enough to train or validate a reliable supervised classifier. Therefore, this implementation does not claim a production-trained ML model. TF-IDF similarity is used as a small-data retrieval aid, and explicit rules drive severity and operational decisions. The evaluation is a smoke test only; its accuracy and macro-F1 must not be interpreted as proof of generalization. Root causes are hypotheses and all decisions require human review. No external LLM is called and no RAG system is claimed. The workflow is agent-style orchestration of local tools, not an autonomous LLM agent.

## Requirements

- Python 3.10 or 3.11 recommended
- Internet only for the first `pip install` (runtime inference uses local bundled data)

## Run on macOS / Linux

```bash
cd Greenko_Maintenance_Ticket_Agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn backend.app:app --reload --port 8000
```

Open **http://127.0.0.1:8000**. API documentation: **http://127.0.0.1:8000/docs**.

## Run on Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn backend.app:app --reload --port 8000
```

## API examples

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

Process a ticket:

```bash
curl -X POST http://127.0.0.1:8000/api/run \
  -H 'Content-Type: application/json' \
  -d '{"asset_type":"wind","description":"Gearbox vibration has increased on turbine WTG-12 with abnormal noise"}'
```

Evaluation:

```bash
curl http://127.0.0.1:8000/api/evaluate
```

## Repository structure

```text
backend/
  app.py                 FastAPI routes and request validation
  triage.py              NLP similarity, rules, tool workflow, evaluation
data/
  maintenance_tickets.csv  Original six-row assignment dataset
frontend/
  index.html             Responsive UI served by FastAPI
tests/
  test_triage.py         Engine tests
  test_api.py            API tests
docs/
  architecture.md        Design and decision flow
  interview_guide.md     Review talking points and likely questions
  evaluation.md          Metric interpretation and limitations
```

## Tests

```bash
pytest -q
```

## Railway deployment

The repository includes a `Procfile` and `railway.json`. To deploy:
1. Push this project to a GitHub repository.
2. In Railway, create a project and deploy from that GitHub repository.
3. Railway should detect the Python project; the start command is `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`.
4. If needed, set the start command explicitly in Railway service settings.
5. Wait for deployment, open the generated public domain, and verify `/api/health` and `/docs`.
6. Add the actual deployment URL to your submission notes only after a successful deployment.

No Railway URL is included because deployment requires your own account and explicit deployment.

## Production improvements

- Collect a larger, diverse and independently labelled ticket dataset; include asset/site context and confirmed outcomes.
- Establish a time/site-aware holdout test set and compare TF-IDF, classical supervised baselines and domain-tuned language models.
- Calibrate confidence and severity against approved site risk matrices and SLAs.
- Add authentication, role-based access, audit logs, rate limits, restricted CORS, monitoring and secure data retention.
- Integrate CMMS/SCADA only through approved, authenticated interfaces; require human approval before work orders or emergency actions.
- Add multilingual and noisy-text handling, duplicate detection and drift monitoring.

## Assignment alignment

This project implements the candidate task: ticket classification, severity scoring, recommended next action and simple agent-style routing. It uses the bundled dummy dataset, provides a Python API and HTML/CSS/JS UI, documents design choices and evaluation, and is structured for Railway deployment.
