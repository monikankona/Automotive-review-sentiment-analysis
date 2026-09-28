# Evaluation Protocol

1. Use a labeled automotive review dataset with one review column and one ground-truth sentiment label.
2. Split into train/validation/test only when model fine-tuning is performed; keep the test set untouched.
3. For the current inference prototype, report label distribution, per-class precision, recall, F1 and confusion matrix only when ground-truth labels are available.
4. Report latency per review and throughput on the actual machine used.
5. For aspect-level evaluation, compare predicted aspect presence and polarity against manually annotated aspect labels.
6. Do not convert model confidence directly into accuracy. Confidence is the model's score for the selected label.
