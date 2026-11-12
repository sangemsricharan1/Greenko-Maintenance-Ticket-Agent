"""Greenko maintenance ticket triage API.

The bundled dataset is intentionally tiny (six examples). This implementation
uses transparent TF-IDF similarity for category suggestions plus explicit,
auditable rules for severity, root-cause hypotheses, actions and routing.
It does not claim a production-trained classifier or autonomous LLM agent.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.triage import TriageEngine

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "maintenance_tickets.csv"
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="Greenko | AI Maintenance Ticket Prioritization",
    description="Explainable NLP-assisted ticket triage and rule-based agent routing demo.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to the deployed frontend origin in production.
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
engine = TriageEngine(DATA_PATH)


class TicketRequest(BaseModel):
    description: str = Field(..., min_length=8, max_length=2000)
    asset_type: str = Field(default="unknown", max_length=40)
    ticket_id: str | None = Field(default=None, max_length=40)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "greenko-maintenance-triage", "dataset_records": engine.dataset_size}


@app.get("/api/demo")
def demo():
    return {
        "case_study": "AI Agent for Maintenance Ticket Prioritization",
        "status": "ready",
        "dataset_records": engine.dataset_size,
        "capabilities": ["category classification", "severity scoring", "root-cause hypothesis", "next-action recommendation", "team routing"],
        "method": "TF-IDF similarity + transparent rule-based decision workflow",
        "limitations": ["Only six labelled sample tickets are supplied", "Recommendations require human verification", "No external LLM is called"],
    }


@app.get("/api/tickets")
def list_tickets():
    return {"count": len(engine.examples), "tickets": engine.examples}


@app.post("/api/run")
def run_ticket(request: TicketRequest):
    try:
        return engine.process(
            description=request.description,
            asset_type=request.asset_type,
            ticket_id=request.ticket_id,
        )
    except Exception as exc:  # Keep internal details out of API responses.
        raise HTTPException(status_code=500, detail="Ticket processing failed. Check server logs.") from exc


@app.get("/api/evaluate")
def evaluate():
    return engine.evaluate()


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
