# Evaluation approach and limitations

The API endpoint `GET /api/evaluate` runs leave-one-out evaluation for category retrieval: one ticket is held out, TF-IDF is fitted on the remaining tickets, and the nearest remaining example's category is predicted. It reports in-sample hybrid-rule matches separately from TF-IDF leave-one-out accuracy and per-class precision/recall/F1. The in-sample rule match is only a coverage check because the rules were designed from the supplied examples; it is not a held-out score. Run the endpoint or `python scripts/evaluate.py` to obtain the exact results for the installed scikit-learn version.

## Interpretation
- Accuracy is the fraction of held-out rows whose category matched the label.
- Precision measures how many predictions for a category were correct.
- Recall measures how many actual examples of a category were recovered.
- F1 is the harmonic mean of precision and recall; macro-F1 gives each category equal weight.

## Critical limitation
There are only six rows and each category appears once. In leave-one-out evaluation, the held-out category is therefore absent from the training fold. The nearest-neighbour method cannot learn a category prototype from the remaining rows. This evaluation is a code-path/sanity check, not a statistically meaningful model benchmark. Do not present the score as production accuracy. The dataset is too small to estimate generalization, class-wise performance or severity calibration.

## Next evaluation steps
1. Obtain a larger, representative dataset with expert-confirmed category and severity labels.
2. Split by time or site to reduce leakage and test realistic future/site generalization.
3. Compare keyword baseline, TF-IDF + Logistic Regression, Linear SVM and suitable transformer embeddings.
4. Report macro-F1, per-class recall, confusion matrix, severity-weighted error and calibration.
5. Review false negatives for Critical/High tickets with maintenance and safety experts.
