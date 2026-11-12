# 1:1 project review – speaking guide

## 45–60 second opening
"This project is an AI-assisted maintenance ticket prioritization prototype for renewable-energy assets. A maintenance ticket usually arrives as free text. The operations team needs to understand what type of issue it is, how urgent it may be, what checks should happen next and which team should receive it. My application accepts an asset type and description, suggests a category, assigns a rule-based severity, provides a probable root-cause hypothesis and next action, and routes it to a priority queue and team. The output is decision support and always requires human verification."

## Explain the architecture
"The browser sends a POST request to FastAPI. Pydantic validates the input. The triage engine first checks transparent category keywords; if no keyword matches, it uses TF-IDF unigram/bigram cosine similarity against the supplied examples. Then severity rules evaluate safety and operating symptoms. The category selects a root-cause hypothesis, action and team. The routing step combines the severity with the team to select P1–P4. The API returns the result and a trace of all five tools."

## Why not train a classifier?
"The supplied dataset has only six labelled tickets, with one example per category. A supervised classifier trained on this would not generalize reliably. I chose a transparent hybrid: rules for explainable category matches and severity, and TF-IDF similarity as a fallback for wording variation. I expose the small-data evaluation but clearly state that it is only a smoke test. With a larger labelled dataset, I would compare Logistic Regression and Linear SVM with a time/site-aware holdout."

## What does the agent do?
"It is a deterministic agent-style workflow, not an LLM agent. It orchestrates five tools: classify_ticket, score_severity, infer_root_cause, recommend_action and route_ticket. Each step uses previous outputs. I did not use an LLM or RAG because the provided data is structured and very small; adding a language model would not solve the lack of labelled evidence."

## Demo sequence
1. Open the app at `http://127.0.0.1:8000`.
2. Select the Wind gearbox sample and click Prioritize ticket.
3. Explain category, severity, likely cause, next action, routing and trace.
4. Try a safety phrase such as "smoke coming from inverter" and show Critical escalation.
5. Click Run evaluation and explain why six examples are not enough for a reliable score.
6. Open `/docs` and show the API endpoints if asked.

## Likely questions and concise answers

**Q: What is the input and output?**
A: Asset type and free-text description go in; category, severity, cause hypothesis, action, routing and evidence come out.

**Q: Why TF-IDF?**
A: It is lightweight, local and explainable for a small offline example set. It represents text with weighted terms and compares vectors using cosine similarity.

**Q: What does severity mean?**
A: It is a first-pass priority recommendation based on explicit trigger phrases and category context, not a certified safety assessment.

**Q: How do you prevent hallucinations?**
A: No generative LLM is used. The output is selected from documented category-specific templates and explicit rules. Root causes are labelled hypotheses, and human review is mandatory.

**Q: Is it RAG?**
A: No. TF-IDF similarity over six ticket examples is not document RAG. There is no document knowledge base or LLM answer generation.

**Q: What are the limitations?**
A: Six rows, no reliable generalization estimate, keyword coverage gaps, no site-specific risk matrix, no CMMS/SCADA integration, and no authentication in this demo.

**Q: What would you improve?**
A: Collect expert-labelled tickets across asset types and sites, validate severity against approved procedures, compare ML baselines, add confidence calibration, audit logs, authentication and human-approved CMMS integration.

## Be honest
Do not claim the app is deployed until you have deployed it and tested the public URL. Do not call the workflow an autonomous LLM agent. Do not call TF-IDF ticket matching RAG. Do not describe the six-row evaluation as proof of model accuracy.
