import json
import argparse
import torch
from transformers import AutoTokenizer
from model import PIIModel
from labels import ID2LABEL, is_pii
import os

def decode_bio(offsets, pred_ids, probs):
    spans = []
    cur_label = None
    cur_start, cur_end = None, None
    scores = []

    for (start, end), pid, p in zip(offsets, pred_ids, probs):
        if start == 0 and end == 0:
            continue
        tag = ID2LABEL[int(pid)]
        if tag == "O":
            if cur_label:
                spans.append({"start": cur_start, "end": cur_end, "label": cur_label, "score": sum(scores) / len(scores)})
            cur_label = None
            scores = []
            continue

        prefix, lab = tag.split("-", 1)

        if prefix == "B":
            if cur_label:
                spans.append({"start": cur_start, "end": cur_end, "label": cur_label, "score": sum(scores) / len(scores)})
            cur_label = lab
            cur_start = start
            cur_end = end
            scores = [p]

        elif prefix == "I":
            if cur_label == lab:
                cur_end = end
                scores.append(p)
            else:
                cur_label = lab
                cur_start = start
                cur_end = end
                scores = [p]

    if cur_label:
        spans.append({"start": cur_start, "end": cur_end, "label": cur_label, "score": sum(scores) / len(scores)})
    return spans

def filter_low_conf(spans, threshold=0.60):
    return [s for s in spans if s["score"] >= threshold and (s["end"] - s["start"]) > 1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    model = PIIModel(args.model_dir)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "pytorch_model.bin")))
    model.to(args.device)
    model.eval()

    results = {}
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            uid, text = obj["id"], obj["text"]
            enc = tokenizer(text, truncation=True, return_offsets_mapping=True, max_length=args.max_length, return_tensors="pt")
            offsets = enc["offset_mapping"][0].tolist()
            ids = enc["input_ids"].to(args.device)
            mask = enc["attention_mask"].to(args.device)
            with torch.no_grad():
                logits = model(ids, mask)[0]
                prob = torch.softmax(logits, dim=-1)
                pred_ids = logits.argmax(dim=-1).cpu().tolist()
                max_p = prob.max(dim=-1).values.cpu().tolist()

            spans = decode_bio(offsets, pred_ids, max_p)
            spans = filter_low_conf(spans)

            ents = [{"start": s["start"], "end": s["end"], "label": s["label"], "pii": bool(is_pii(s["label"]))} for s in spans]
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
