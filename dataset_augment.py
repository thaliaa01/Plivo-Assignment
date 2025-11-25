import json
import random
import os
from faker import Faker

fake = Faker()
random.seed(42)
Faker.seed(42)

TRAIN_SIZE = 800
DEV_SIZE = 150
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

ENTITY_TYPES = [
    "CREDIT_CARD",
    "PHONE",
    "EMAIL",
    "PERSON_NAME",
    "DATE",
    "CITY",
    "LOCATION",
]

def num_to_words(n):
    mapping = {
        "0": "zero", "1": "one", "2": "two", "3": "three",
        "4": "four", "5": "five", "6": "six", "7": "seven",
        "8": "eight", "9": "nine"
    }
    return " ".join(mapping[d] for d in str(n))

def generate_email():
    username = fake.user_name().replace(".", " ")
    return f"{username} at gmail dot com"

def generate_phone():
    num = "".join(str(random.randint(0, 9)) for _ in range(10))
    return num_to_words(num)

def generate_credit_card():
    num = "".join(str(random.randint(0, 9)) for _ in range(16))
    return num_to_words(num)

def generate_date():
    date = fake.date()
    y, m, d = date.split("-")
    return f"{num_to_words(d)} {fake.month_name().lower()} {num_to_words(y)}"

def generate_entity(label):
    if label == "CREDIT_CARD": return generate_credit_card()
    if label == "PHONE": return generate_phone()
    if label == "EMAIL": return generate_email()
    if label == "PERSON_NAME": return fake.name().lower()
    if label == "DATE": return generate_date()
    if label == "CITY": return fake.city().lower()
    if label == "LOCATION": return fake.address().replace("\n", " ").lower()
    return ""

def add_stt_noise(words):
    noisy = []
    for w in words:
        r = random.random()
        if r < 0.05:
            noisy.append("uh")
            noisy.append(w)
        elif r < 0.10 and len(w) > 4:
            noisy.append(w[:-1])
        elif r < 0.15:
            noisy.append(w)
            noisy.append(w)
        else:
            noisy.append(w)
    return noisy

def generate_sample(i):
    num_ents = random.randint(1, 3)
    labels = random.sample(ENTITY_TYPES, num_ents)
    template_tokens = []
    entities = []

    for label in labels:
        ent_text = generate_entity(label)
        ent_tokens = ent_text.split()
        placeholder = f"<<ENT{i}_{label}>>"
        template = ["the", "user", "mentioned", placeholder, "in", "the", "conversation"]
        template_tokens.extend(template)
        entities.append({"placeholder": placeholder, "tokens": ent_tokens, "label": label})

    noisy_tokens = add_stt_noise(template_tokens)
    noisy_text = " ".join(noisy_tokens)
    final_text = noisy_text
    final_entities = []

    for ent in entities:
        placeholder = ent["placeholder"]
        ent_string = " ".join(ent["tokens"])
        idx = final_text.find(placeholder)
        if idx == -1:
            continue
        before = final_text[:idx]
        after = final_text[idx + len(placeholder):]
        final_text = before + ent_string + after

    spans = []
    for ent in entities:
        ent_string = " ".join(ent["tokens"])
        label = ent["label"]
        start = final_text.find(ent_string)
        if start == -1:
            continue
        end = start + len(ent_string)
        spans.append({"start": start, "end": end, "label": label})

    return {"id": f"utt_{i:05d}", "text": final_text, "entities": spans}

def write_jsonl(path, samples):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

def main():
    train = [generate_sample(i) for i in range(1, TRAIN_SIZE + 1)]
    dev = [generate_sample(i) for i in range(9001, 9001 + DEV_SIZE)]
    write_jsonl(os.path.join(DATA_DIR, "train.jsonl"), train)
    write_jsonl(os.path.join(DATA_DIR, "dev.jsonl"), dev)
    print("Done:", len(train), "train,", len(dev), "dev")

if __name__ == "__main__":
    main()
