import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import PIIDataset, collate_batch
from labels import ALL_LABELS
from model import PIIModel

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-cased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()

def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total = 0.0
    for batch in loader:
        ids = torch.tensor(batch["input_ids"], device=device)
        mask = torch.tensor(batch["attention_mask"], device=device)
        labels = torch.tensor(batch["labels"], device=device)
        logits = model(ids, mask)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / len(loader)

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_data = PIIDataset(args.train, tokenizer, ALL_LABELS, max_length=args.max_length)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id),
    )
    model = PIIModel(args.model_name).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, args.device)
        print(f"Epoch {ep+1}/{args.epochs} | Loss: {avg_loss:.4f}")

    model.encoder.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "pytorch_model.bin"))
    print("Model saved.")

if __name__ == "__main__":
    main()
