# Troubleshooting

## Config errors

### Symptom

`ConfigurationError` raised at startup.

### Fix

1. Verify config file path.
2. Confirm required keys in `dataset` section.
3. Ensure `test_size` is between 0 and 1.

## CSV parsing errors

### Symptom

Rows fail with numeric conversion errors.

### Fix

1. Ensure all feature columns are numeric.
2. Ensure `label_column` exists and is non-empty for each row.

## sklearn model import failures

### Symptom

Import error when `model.kind: sklearn` is used.

### Fix

Install optional ML dependencies:

```bash
pip install -e .[ml]
```

## joblib artifact failures

### Symptom

Cannot save or load `.joblib` model artifacts.

### Fix

Install ML extras and confirm model artifact extension and type are aligned.

## Low evaluation metrics

### Symptom

Accuracy/precision/recall below expectations.

### Fix

1. Tune classifier configuration.
2. Inspect class balance.
3. Add feature engineering in a custom preprocessing step.
