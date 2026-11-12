# Architecture and design decisions

## Business problem
Maintenance requests across solar, wind, hydro and grid assets need consistent first-line triage. The prototype converts a free-text issue description into a structured decision-support record so the operations team can review urgency, likely issue type, suggested checks and destination team.

## Input and output
**Input:** asset type (optional), ticket ID (optional), and a plain-language description.

**Output:** category, method/confidence, severity and reasons, probable root-cause hypothesis, next action, priority queue, team, similar-example evidence and a step-by-step tool trace.

## Processing flow
1. Validate request using Pydantic.
2. Normalize asset type and text.
3. Apply transparent category keyword rules. If no rule matches, compute TF-IDF unigram/bigram cosine similarity against the six labelled examples. If similarity is weak, use `general_maintenance`.
4. Apply severity triggers. Critical safety terms override other levels; high-risk equipment symptoms become High; operational/compliance issues without immediate safety terms default to Medium; Low is reserved for explicit minor/cosmetic language.
5. Use the predicted category to select a root-cause hypothesis, recommended next action and responsible team.
6. Route by severity queue (P1–P4). Critical tickets are escalated to the site control room and emergency maintenance lead.
7. Return structured output and an auditable trace. Human review is always required.

## Why this approach
The assignment dataset has six rows and one example per category. A conventional supervised model would be misleading and highly prone to overfitting. TF-IDF similarity is lightweight, local, reproducible and inspectable; keyword and severity rules make the decision path transparent. A larger labelled dataset is required before training and benchmarking a supervised model.

## Agent-style workflow
The workflow orchestrates five local tools in sequence: classification, severity scoring, root-cause hypothesis, action recommendation and routing. Outputs from earlier steps feed later steps. This is a simple deterministic tool workflow; it is not an autonomous LLM agent. No LLM or RAG is used because the bundled task can be demonstrated with structured examples and explicit rules.

## Safety and safeguards
- Root cause is worded as a possibility, not a confirmed diagnosis.
- Critical safety language triggers immediate escalation guidance.
- The response explicitly requires human review.
- Unknown categories use a general maintenance fallback.
- No external API keys or secrets are required.
- In production, replace permissive CORS, add authentication/authorization and enforce site-approved emergency procedures.
