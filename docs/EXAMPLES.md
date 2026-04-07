# Examples

## 1. Minimal train flow

```bash
ai-classifier train --config examples/configs/template_config.yaml
```

## 2. Predict with generated model

```bash
ai-classifier predict --model artifacts/model.json --input examples/data/predict.csv --output artifacts/predictions.csv
```

## 3. Evaluate generated model

```bash
ai-classifier evaluate --model artifacts/model.json --input examples/data/train.csv --label-column label --output artifacts/eval.json
```

## 4. Predict with starter pre-trained artifact

```bash
ai-classifier predict --model examples/data/pretrained_model.json --input examples/data/predict.csv --output artifacts/predictions_from_pretrained.csv
```

## 5. Python API usage

```python
from ai_classifier.core.config import PipelineConfig
from ai_classifier.training.pipeline import TrainingPipeline

config = PipelineConfig.from_yaml("examples/configs/template_config.yaml")
report = TrainingPipeline(config).run()
print(report)
```
