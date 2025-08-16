# multi_label_news_bert.py
import os
import json
import random
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# -----------------------
# CONFIG
# -----------------------
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
DATA_CSV = "news_multilabel.csv"   # <-- your dataset file

# -----------------------
# REPRODUCIBILITY
# -----------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

# -----------------------
# 1) LOAD DATA
# -----------------------
df = pd.read_csv(DATA_CSV)   # expects columns: "text" and "labels"
assert "text" in df.columns and "labels" in df.columns, "CSV must have 'text' and 'labels' columns"

# Clean / drop nulls
df = df.dropna(subset=["text", "labels"]).reset_index(drop=True)

# Split labels (assume '|' separator)
df["label_list"] = df["labels"].apply(lambda s: [t.strip() for t in str(s).split("|") if t.strip()])

# Build label vocabulary
all_labels = sorted({label for row in df["label_list"] for label in row})
label2idx = {label: i for i, label in enumerate(all_labels)}
idx2label = {i: l for l, i in label2idx.items()}

NUM_LABELS = len(all_labels)
print(f"Num examples: {len(df)}, Num labels: {NUM_LABELS}")
print("Some labels:", all_labels[:10])

# Convert to multi-hot vectors
def labels_to_multihot(labels, label2idx):
    vec = np.zeros(len(label2idx), dtype=np.float32)
    for l in labels:
        if l in label2idx:
            vec[label2idx[l]] = 1.0
    return vec

df["multi_hot"] = df["label_list"].apply(lambda L: labels_to_multihot(L, label2idx))

# Train-val-test split
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED, shuffle=True)
val_df, test_df  = train_test_split(temp_df, test_size=0.5, random_state=SEED, shuffle=True)

print("Splits:", len(train_df), len(val_df), len(test_df))

# -----------------------
# 2) TOKENIZATION + DATASET
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class NewsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[np.ndarray], tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32)
        }
        return item

train_ds = NewsDataset(train_df["text"].tolist(), train_df["multi_hot"].tolist(), tokenizer)
val_ds   = NewsDataset(val_df["text"].tolist(), val_df["multi_hot"].tolist(), tokenizer)
test_ds  = NewsDataset(test_df["text"].tolist(), test_df["multi_hot"].tolist(), tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------
# 3) MODEL
# -----------------------
class BertForMultiLabel(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout_prob: float = 0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # use pooler_output if available, else CLS token
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

model = BertForMultiLabel(MODEL_NAME, NUM_LABELS)
model.to(DEVICE)

# -----------------------
# 4) LOSS, OPTIMIZER, SCHEDULER
# -----------------------
loss_fn = nn.BCEWithLogitsLoss()  # logits -> BCE
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

# -----------------------
# 5) TRAIN / EVAL utilities
# -----------------------
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def evaluate(model, data_loader, threshold=0.5):
    model.eval()
    all_labels = []
    all_preds = []
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds)
            losses.append(loss.item())

    all_labels = np.vstack(all_labels)
    all_preds  = np.vstack(all_preds)
    avg_loss = float(np.mean(losses))

    # metrics
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average="micro", zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    # Hamming accuracy (1 - hamming loss)
    acc = accuracy_score(all_labels, all_preds)   # exact match accuracy (strict)
    return {"loss": avg_loss, "micro_f1": f1_micro, "micro_precision": p_micro, "micro_recall": r_micro,
            "macro_f1": f1_macro, "macro_precision": p_macro, "macro_recall": r_macro, "exact_acc": acc}

# -----------------------
# 6) TRAINING LOOP
# -----------------------
best_micro_f1 = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    losses = []
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
    for batch in loop:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        loop.set_postfix(loss=loss.item())

    train_loss = float(np.mean(losses))
    val_metrics = evaluate(model, val_loader, threshold=0.5)
    print(f"Epoch {epoch} | Train loss {train_loss:.4f} | Val micro-F1 {val_metrics['micro_f1']:.4f} | Val macro-F1 {val_metrics['macro_f1']:.4f}")

    # save best
    if val_metrics["micro_f1"] > best_micro_f1:
        best_micro_f1 = val_metrics["micro_f1"]
        torch.save({
            "model_state_dict": model.state_dict(),
            "label2idx": label2idx,
            "idx2label": idx2label,
            "tokenizer_name": MODEL_NAME
        }, "best_model.pth")
        print(f"> Saved best model (micro_f1={best_micro_f1:.4f})")

# -----------------------
# 7) TEST EVALUATION
# -----------------------
checkpoint = torch.load("best_model.pth", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
test_metrics = evaluate(model, test_loader, threshold=0.5)
print("Test results:", test_metrics)

# -----------------------
# 8) INFERENCE function
# -----------------------
def predict_texts(model, tokenizer, texts: List[str], threshold=0.5, max_len=MAX_LEN):
    model.eval()
    ds = NewsDataset(texts, [np.zeros(NUM_LABELS)]*len(texts), tokenizer, max_len=max_len)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    all_probs = np.vstack(all_probs)
    results = []
    for prob_row in all_probs:
        idxs = np.where(prob_row >= threshold)[0].tolist()
        labels = [idx2label[i] for i in idxs]
        results.append({"labels": labels, "probs": prob_row.tolist()})
    return results

# Example usage:
sample_texts = [
    "Government unveils economic stimulus package and tech grants to boost startups.",
    "Local team wins the championship in a thrilling overtime victory."
]
preds = predict_texts(model, tokenizer, sample_texts, threshold=0.4)
for t, p in zip(sample_texts, preds):
    print("TEXT:", t)
    print("PRED:", p)
