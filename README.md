# PII NER Assignment

This repository implements a token-level Named Entity Recognition (NER) model that detects and classifies PII entities in noisy speech-to-text (STT) transcripts.
The model assigns BIO tags token-wise, converts them to character spans, and marks each detected entity as **PII** or **non-PII** based on the assignment rules.

---

## Setup

```bash
pip install -r requirements.txt
```

(Optional) Create a virtual environment before installing dependencies.

---

## Train

```bash
python src/train.py \
  --model_name distilbert-base-cased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out
```

This will train the model and save the trained weights + tokenizer into the `out/` directory.

---

## Predict

```bash
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json
```

This generates predictions in the required JSON output format.

---

## Evaluate

### Dev set

```bash
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json
```

### (Optional) Stress test set

```bash
python src/predict.py \
  --model_dir out \
  --input data/stress.jsonl \
  --output out/stress_pred.json
```

```bash
python src/eval_span_f1.py \
  --gold data/stress.jsonl \
  --pred out/stress_pred.json
```

This reports span-level metrics including precision, recall, F1-score per entity, macro-F1, and PII-only vs non-PII evaluation.

---

## Measure Latency

```bash
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50
```

Latency is evaluated per single-utterance prediction at batch size = 1. Metrics include mean, p50, and p95 inference time.

---

## (Optional) Synthetic Dataset Generation

```bash
python dataset_generator.py
```

This regenerates `train.jsonl` and `dev.jsonl` with synthetic, noisy STT-style labeled examples.

---

## Output Format Example

```json
{
  "utt_0012": [
    { "start": 3, "end": 19, "label": "CREDIT_CARD", "pii": true },
    { "start": 63, "end": 77, "label": "PERSON_NAME", "pii": true },
    { "start": 81, "end": 105, "label": "EMAIL", "pii": true }
  ]
}




