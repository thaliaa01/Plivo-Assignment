import json
import argparse
from collections import defaultdict
from labels import is_pii

def load_gold(path):
    gold = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            spans = [(e["start"], e["end"], e["label"]) for e in obj.get("entities", [])]
            gold[obj["id"]] = spans
    return gold

def load_pred(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pred = {}
    for uid, ents in data.items():
        pred[uid] = [(e["start"], e["end"], e["label"]) for e in ents]
    return pred

def metrics(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return p, r, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    gold = load_gold(args.gold)
    pred = load_pred(args.pred)

    label_set = set(lab for spans in gold.values() for _, _, lab in spans)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for uid in gold:
        g = set(gold.get(uid, []))
        p = set(pred.get(uid, []))
        for span in p:
            if span in g: tp[span[2]] += 1
            else: fp[span[2]] += 1
        for span in g:
            if span not in p: fn[span[2]] += 1

    print("Per-entity metrics:")
    total_f1 = 0
    for lab in sorted(label_set):
        p, r, f1 = metrics(tp[lab], fp[lab], fn[lab])
        total_f1 += f1
        print(f"{lab:15s} P={p:.3f} R={r:.3f} F1={f1:.3f}")

    print("\nMacro-F1:", round(total_f1 / len(label_set), 3))

    pii_tp = pii_fp = pii_fn = 0
    non_tp = non_fp = non_fn = 0

    for uid in gold:
        g = gold.get(uid, [])
        p = pred.get(uid, [])

        gold_p = {(s, e, "PII") for s, e, lab in g if is_pii(lab)}
        pred_p = {(s, e, "PII") for s, e, lab in p if is_pii(lab)}

        gold_n = {(s, e, "NON") for s, e, lab in g if not is_pii(lab)}
        pred_n = {(s, e, "NON") for s, e, lab in p if not is_pii(lab)}

        for x in pred_p:
            if x in gold_p: pii_tp += 1
            else: pii_fp += 1
        for x in gold_p:
            if x not in pred_p: pii_fn += 1

        for x in pred_n:
            if x in gold_n: non_tp += 1
            else: non_fp += 1
        for x in gold_n:
            if x not in pred_n: non_fn += 1

    p, r, f1 = metrics(pii_tp, pii_fp, pii_fn)
    print(f"\nPII Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")

    p2, r2, f12 = metrics(non_tp, non_fp, non_fn)
    print(f"Non-PII Precision={p2:.3f} Recall={r2:.3f} F1={f12:.3f}")

if __name__ == "__main__":
    main()
