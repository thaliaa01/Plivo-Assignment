import json
import time
import argparse
import statistics
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])

    for _ in range(3):
        enc = tokenizer(texts[0], return_tensors="pt", truncation=True, max_length=args.max_length)
        with torch.no_grad():
            _ = model(**{k: v.to(args.device) for k, v in enc.items()})

    times = []
    for i in range(args.runs):
        t = texts[i % len(texts)]
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=args.max_length)
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(**{k: v.to(args.device) for k, v in enc.items()})
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times_sorted = sorted(times)
    p50 = statistics.median(times)
    p95 = times_sorted[int(0.95 * len(times)) - 1]

    print(f"Mean: {statistics.mean(times):.2f} ms")
    print(f"Std: {statistics.stdev(times):.2f} ms")
    print(f"p50: {p50:.2f} ms")
    print(f"p95: {p95:.2f} ms")

if __name__ == "__main__":
    main()
