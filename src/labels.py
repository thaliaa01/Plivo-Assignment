ALL_LABELS = [
    "O",
    "B-CREDIT_CARD", "I-CREDIT_CARD",
    "B-PHONE", "I-PHONE",
    "B-EMAIL", "I-EMAIL",
    "B-PERSON_NAME", "I-PERSON_NAME",
    "B-DATE", "I-DATE",
    "B-CITY", "I-CITY",
    "B-LOCATION", "I-LOCATION",
]

PII_CATEGORIES = {
    "CREDIT_CARD",
    "PHONE",
    "EMAIL",
    "PERSON_NAME",
    "DATE",
}

LABEL2ID = {lbl: idx for idx, lbl in enumerate(ALL_LABELS)}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def is_pii(label: str) -> bool:
    return label in PII_CATEGORIES
