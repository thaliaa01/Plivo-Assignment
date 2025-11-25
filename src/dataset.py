import json
from torch.utils.data import Dataset

class PIIDataset(Dataset):
    def __init__(self, path, tokenizer, label_list, max_length=256):
        self.items = []
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.max_length = max_length

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj["text"]
                entities = obj.get("entities", [])
                char_tags = ["O"] * len(text)

                for e in entities:
                    s, e2, lab = e["start"], e["end"], e["label"]
                    if 0 <= s < e2 <= len(text):
                        char_tags[s] = f"B-{lab}"
                        for i in range(s + 1, e2):
                            char_tags[i] = f"I-{lab}"

                enc = tokenizer(
                    text,
                    truncation=True,
                    return_offsets_mapping=True,
                    max_length=self.max_length,
                    add_special_tokens=True,
                )

                offsets = enc["offset_mapping"]
                token_ids = enc["input_ids"]
                attn = enc["attention_mask"]
                bio_tags = []

                for start, end in offsets:
                    if start == end:
                        bio_tags.append("O")
                    else:
                        bio_tags.append(char_tags[start] if start < len(char_tags) else "O")

                label_ids = [self.label2id.get(t, self.label2id["O"]) for t in bio_tags]

                self.items.append({
                    "id": obj["id"],
                    "input_ids": token_ids,
                    "attention_mask": attn,
                    "labels": label_ids,
                    "offsets": offsets,
                    "text": text,
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def collate_batch(batch, pad_token_id, label_pad=-100):
    max_len = max(len(x["input_ids"]) for x in batch)
    def pad(seq, val): return seq + [val] * (max_len - len(seq))

    return {
        "input_ids": [pad(x["input_ids"], pad_token_id) for x in batch],
        "attention_mask": [pad(x["attention_mask"], 0) for x in batch],
        "labels": [pad(x["labels"], label_pad) for x in batch],
    }
