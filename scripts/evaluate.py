from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.triage import TriageEngine  # noqa: E402

engine = TriageEngine(ROOT / "data" / "maintenance_tickets.csv")
print(json.dumps(engine.evaluate(), indent=2))
