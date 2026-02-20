#!/usr/bin/env python3
"""
Task 1 – Multilingual ADE detection (SMM4H 2026, XLM-RoBERTa-base)
Train on DE+EN, zero-shot eval on RU, translate EN->RU for task 1.3.3
"""

import os
import sys
import json
import time
import warnings
import random
import re
from collections import Counter

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix, roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from deep_translator import GoogleTranslator
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw", "SMM4H_2026_Task_1")
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
RESULT_DIR = os.path.join(BASE_DIR, "outputs", "results")

for d in [DATA_PROC, FIG_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

SEED = 42
TRAIN_LANGS = ["de", "en"]
ZEROSHOT_LANG = "ru"
ALL_LANGS = TRAIN_LANGS + [ZEROSHOT_LANG]

MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
VAL_RATIO = 0.15
TRANSLATE_N = 500

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
print(f"Using device: {DEVICE}", flush=True)


# --- 1.1: Data loading, splits, translation ---

def load_raw_data():
    train_path = os.path.join(DATA_RAW, "train_data_SMM4H_2026_Task_1.csv")
    dev_path = os.path.join(DATA_RAW, "dev_data_SMM4H_2026_Task_1.csv")
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    train_df["label"] = train_df["label"].astype(int)
    dev_df["label"] = dev_df["label"].astype(int)
    return train_df, dev_df


def prepare_splits(train_df, dev_df):
    """dev=test, train split 85/15 for DE+EN, RU is test-only."""
    splits = {}
    for lang in ALL_LANGS:
        lang_train = train_df[train_df["language"] == lang].reset_index(drop=True)
        lang_test = dev_df[dev_df["language"] == lang].reset_index(drop=True)

        if lang in TRAIN_LANGS:
            new_train, new_val = train_test_split(
                lang_train,
                test_size=VAL_RATIO,
                stratify=lang_train["label"],
                random_state=SEED,
            )
            splits[lang] = {
                "train": new_train.reset_index(drop=True),
                "val": new_val.reset_index(drop=True),
                "test": lang_test,
            }
        else:
            splits[lang] = {
                "train": pd.DataFrame(columns=lang_train.columns),
                "val": pd.DataFrame(columns=lang_train.columns),
                "test": lang_test,
            }
    return splits


def translate_english_to_russian(en_train_df, n=TRANSLATE_N):
    """Translate EN subset -> RU via Google Translate (cached)."""
    cache_path = os.path.join(DATA_PROC, "en_translated_to_ru.csv")
    if os.path.exists(cache_path):
        print(f"  Loading cached translations from {cache_path}", flush=True)
        return pd.read_csv(cache_path)

    pos = en_train_df[en_train_df["label"] == 1]
    neg = en_train_df[en_train_df["label"] == 0]

    n_pos = min(len(pos), n // 5)
    n_neg = min(len(neg), n - n_pos)

    sample_pos = pos.sample(n=n_pos, random_state=SEED)
    sample_neg = neg.sample(n=n_neg, random_state=SEED)
    sample = pd.concat([sample_pos, sample_neg]).reset_index(drop=True)

    print(f"  Translating {len(sample)} English samples to Russian...", flush=True)
    translator = GoogleTranslator(source="en", target="ru")
    translated_texts = []
    for text in tqdm(sample["text"].tolist(), desc="Translating"):
        try:
            result = translator.translate(text[:4999])
            translated_texts.append(result if result else text)
        except Exception as e:
            translated_texts.append(text)
            if "429" in str(e) or "Too Many" in str(e):
                time.sleep(10)
            else:
                time.sleep(1)

    translated_df = sample.copy()
    translated_df["text_original"] = sample["text"]
    translated_df["text"] = translated_texts
    translated_df["language"] = "ru"
    translated_df["origin"] = "translated_en"
    translated_df.to_csv(cache_path, index=False)
    print(f"  Saved translations to {cache_path}", flush=True)
    return translated_df


def report_split_sizes(splits, translated_df):
    print("\n" + "=" * 60, flush=True)
    print("TASK 1.1 – Dataset Sizes and Language Distributions")
    print("=" * 60)
    rows = []
    for lang in ALL_LANGS:
        s = splits[lang]
        for part in ["train", "val", "test"]:
            df = s[part]
            n = len(df)
            n_pos = int(df["label"].sum()) if n > 0 else 0
            rows.append({
                "Language": lang.upper(),
                "Split": part,
                "Total": n,
                "Positive (1)": n_pos,
                "Negative (0)": n - n_pos,
                "Pos %": f"{n_pos / n * 100:.1f}" if n > 0 else "–",
            })
    rows.append({
        "Language": "RU (translated EN)",
        "Split": "train",
        "Total": len(translated_df),
        "Positive (1)": int(translated_df["label"].sum()),
        "Negative (0)": len(translated_df) - int(translated_df["label"].sum()),
        "Pos %": f"{translated_df['label'].sum() / len(translated_df) * 100:.1f}",
    })
    size_df = pd.DataFrame(rows)
    print(size_df.to_string(index=False), flush=True)
    size_df.to_csv(os.path.join(RESULT_DIR, "split_sizes.csv"), index=False)
    return size_df


# --- 1.2: EDA ---

def exploratory_analysis(splits, translated_df):
    print("\n" + "=" * 60, flush=True)
    print("TASK 1.2 – Multilingual Data Exploration and Analysis")
    print("=" * 60)

    # label distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    lang_labels = []
    for lang in ALL_LANGS:
        df = splits[lang]["train"] if len(splits[lang]["train"]) > 0 else splits[lang]["test"]
        lang_labels.append({
            "Language": lang.upper(),
            "Negative (0)": int((df["label"] == 0).sum()),
            "Positive (1)": int((df["label"] == 1).sum()),
        })
    label_df = pd.DataFrame(lang_labels).set_index("Language")
    label_df.plot(kind="bar", ax=ax, color=["steelblue", "coral"])
    ax.set_title("Label Distribution per Language (ADE detection)")
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.legend(title="Label")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "label_distribution.png"), dpi=150)
    plt.close()
    print("  Saved: label_distribution.png", flush=True)

    # dataset sizes
    fig, ax = plt.subplots(figsize=(10, 6))
    size_data = []
    for lang in ALL_LANGS:
        for part in ["train", "val", "test"]:
            size_data.append({
                "Language": lang.upper(),
                "Split": part.capitalize(),
                "Count": len(splits[lang][part]),
            })
    size_plot_df = pd.DataFrame(size_data)
    sns.barplot(data=size_plot_df, x="Language", y="Count", hue="Split", ax=ax)
    ax.set_title("Dataset Size by Language and Split")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "dataset_sizes.png"), dpi=150)
    plt.close()
    print("  Saved: dataset_sizes.png", flush=True)

    # text lengths
    n_langs = len(ALL_LANGS)
    fig, axes = plt.subplots(1, n_langs, figsize=(7 * n_langs, 5))
    for idx, lang in enumerate(ALL_LANGS):
        df = splits[lang]["train"] if len(splits[lang]["train"]) > 0 else splits[lang]["test"]
        lengths = df["text"].astype(str).str.len()
        axes[idx].hist(lengths, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        axes[idx].set_title(f"{lang.upper()} – Text Length (chars)")
        axes[idx].set_xlabel("Characters")
        axes[idx].set_ylabel("Count")
        stats = f"Mean: {lengths.mean():.0f}\nMedian: {lengths.median():.0f}\nMax: {lengths.max():.0f}"
        axes[idx].text(0.95, 0.95, stats, transform=axes[idx].transAxes,
                       va="top", ha="right", fontsize=9,
                       bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.suptitle("Text Length Distribution per Language", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "text_length_histogram.png"), dpi=150)
    plt.close()
    print("  Saved: text_length_histogram.png", flush=True)

    # top-20 words per language
    fig, axes = plt.subplots(1, n_langs, figsize=(7 * n_langs, 6))
    for idx, lang in enumerate(ALL_LANGS):
        df = splits[lang]["train"] if len(splits[lang]["train"]) > 0 else splits[lang]["test"]
        words = " ".join(df["text"].dropna().astype(str).tolist()).lower().split()
        words = [w for w in words if len(w) > 2 and w.isalpha()]
        freq = Counter(words).most_common(20)
        if freq:
            w, c = zip(*freq)
            axes[idx].barh(list(reversed(w)), list(reversed(c)), color="steelblue")
        axes[idx].set_title(f"{lang.upper()} – Top 20 Words")
        axes[idx].set_xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "word_frequency.png"), dpi=150)
    plt.close()
    print("  Saved: word_frequency.png", flush=True)

    # token lengths (XLM-R tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    fig, axes = plt.subplots(1, n_langs, figsize=(7 * n_langs, 5))
    for idx, lang in enumerate(ALL_LANGS):
        df = splits[lang]["train"] if len(splits[lang]["train"]) > 0 else splits[lang]["test"]
        sample_texts = df["text"].dropna().astype(str).tolist()[:2000]
        tok_lens = [len(tokenizer.encode(t, truncation=False)) for t in sample_texts]
        axes[idx].hist(tok_lens, bins=50, color="teal", edgecolor="white", alpha=0.8)
        axes[idx].axvline(x=MAX_LEN, color="red", linestyle="--", label=f"max_len={MAX_LEN}")
        over = sum(1 for tl in tok_lens if tl > MAX_LEN)
        axes[idx].set_title(f"{lang.upper()} – Tokens ({over}/{len(tok_lens)} > {MAX_LEN})")
        axes[idx].set_xlabel("Tokens")
        axes[idx].legend()
    plt.suptitle("XLM-RoBERTa Token Length Distribution", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "token_length_distribution.png"), dpi=150)
    plt.close()
    print("  Saved: token_length_distribution.png", flush=True)

    # sample inspection
    print("\n--- Manual Sample Inspection ---", flush=True)
    for lang in ALL_LANGS:
        df = splits[lang]["train"] if len(splits[lang]["train"]) > 0 else splits[lang]["test"]
        print(f"\n  [{lang.upper()}] Positive samples (ADE, label=1):")
        pos = df[df["label"] == 1]
        for _, row in pos.head(3).iterrows():
            print(f"    + {str(row['text'])[:200]}")
        print(f"  [{lang.upper()}] Negative samples (no ADE, label=0):")
        neg = df[df["label"] == 0]
        for _, row in neg.head(3).iterrows():
            print(f"    - {str(row['text'])[:200]}")

    # translation quality check
    print("\n--- Translation Quality Inspection (EN -> RU) ---", flush=True)
    if "text_original" in translated_df.columns:
        sample = translated_df.head(10)
        for _, row in sample.iterrows():
            print(f"  EN: {str(row['text_original'])[:150]}")
            print(f"  RU: {str(row['text'])[:150]}")
            print()

    print("--- Notes ---", flush=True)
    print("  DE=lifeline.de forum, EN=Twitter, RU=RuDReC drug reviews")
    print("  Training: DE+EN, Zero-shot eval: RU (Cyrillic script)")
    print("  RU uses Cyrillic => real cross-script transfer test", flush=True)


# --- 1.3: Training ---

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def tokenize(texts, tokenizer, max_len=MAX_LEN):
    return tokenizer(texts, truncation=True, padding="max_length",
                     max_length=max_len, return_tensors="pt")


def make_dataset(df, tokenizer):
    texts = df["text"].fillna("").astype(str).tolist()
    labels = df["label"].tolist()
    enc = tokenize(texts, tokenizer)
    return TextDataset(enc, labels)


def compute_weights(labels):
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=np.array(labels))
    return torch.tensor(cw, dtype=torch.float32)


def load_model_from_checkpoint(save_path, device=DEVICE):
    weights_path = os.path.join(save_path, "model.safetensors")
    weights_path2 = os.path.join(save_path, "pytorch_model.bin")
    if os.path.exists(weights_path) or os.path.exists(weights_path2):
        print(f"  Loading checkpoint from {save_path}...", flush=True)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(save_path, num_labels=2)
            model = model.to(device)
            return model, True
        except Exception as e:
            print(f"  Failed to load checkpoint: {e}", flush=True)
    return None, False


def train_model(train_dataset, val_dataset, model_name=MODEL_NAME,
                epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                class_weights=None, device=DEVICE, save_path=None):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if class_weights is not None:
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # val
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labs = batch["labels"].to(device)
                out = model(input_ids=ids, attention_mask=mask)
                val_loss += loss_fn(out.logits, labs).item()
                all_preds.extend(torch.argmax(out.logits, dim=-1).cpu().numpy())
                all_labels.extend(labs.cpu().numpy())
        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_f1"].append(val_f1)

        print(f"  Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Macro-F1: {val_f1:.4f}", flush=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        print(f"  Model saved to {save_path}", flush=True)

    return model, history


def evaluate_model(model, dataset, device=DEVICE, return_preds=False):
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=-1)[:, 1]
            preds = torch.argmax(out.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = {
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "binary_f1": f1_score(all_labels, all_preds, average="binary", zero_division=0),
        "precision": precision_score(all_labels, all_preds, average="binary", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="binary", zero_division=0),
        "accuracy": accuracy_score(all_labels, all_preds),
    }
    try:
        metrics["auc_roc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["auc_roc"] = float("nan")

    if return_preds:
        return metrics, all_preds, all_labels
    return metrics


def run_all_experiments(splits, translated_df, tokenizer):
    results = []
    models = {}
    histories = {}

    # datasets
    de_train_ds = make_dataset(splits["de"]["train"], tokenizer)
    de_val_ds = make_dataset(splits["de"]["val"], tokenizer)
    en_train_ds = make_dataset(splits["en"]["train"], tokenizer)
    en_val_ds = make_dataset(splits["en"]["val"], tokenizer)

    # combined DE+EN
    multi_train_df = pd.concat([splits["de"]["train"], splits["en"]["train"]]).reset_index(drop=True)
    multi_val_df = pd.concat([splits["de"]["val"], splits["en"]["val"]]).reset_index(drop=True)
    multi_train_ds = make_dataset(multi_train_df, tokenizer)
    multi_val_ds = make_dataset(multi_val_df, tokenizer)

    # translated data for 1.3.3c
    translated_ds = make_dataset(translated_df, tokenizer)

    # test sets
    test_sets = {}
    for lang in ALL_LANGS:
        test_sets[lang] = make_dataset(splits[lang]["test"], tokenizer)

    def eval_on_lang(model, model_name, lang):
        m = evaluate_model(model, test_sets[lang])
        m["model"] = model_name
        m["test_lang"] = lang.upper()
        results.append(m)
        print(f"  -> {lang.upper()} test: Macro-F1={m['macro_f1']:.4f}  "
              f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"Acc={m['accuracy']:.4f}", flush=True)

    # -- 1.3.1: monolingual --
    print("\n" + "=" * 60, flush=True)
    print("Experiment 1: Monolingual German (Task 1.3.1)")
    print("=" * 60)
    checkpoint_path = os.path.join(MODEL_DIR, "mono_de")
    model_de, loaded = load_model_from_checkpoint(checkpoint_path)
    if loaded:
        hist_de = {"train_loss": [], "val_loss": [], "val_f1": []}
        print("  Skipping training (checkpoint loaded)", flush=True)
    else:
        de_cw = compute_weights(splits["de"]["train"]["label"].tolist())
        model_de, hist_de = train_model(
            de_train_ds, de_val_ds, class_weights=de_cw,
            save_path=checkpoint_path)
    models["mono_de"] = model_de
    histories["mono_de"] = hist_de
    eval_on_lang(model_de, "Monolingual DE", "de")

    # mono EN
    print("\n" + "=" * 60, flush=True)
    print("Experiment 2: Monolingual English (Task 1.3.1)")
    print("=" * 60)
    checkpoint_path = os.path.join(MODEL_DIR, "mono_en")
    model_en, loaded = load_model_from_checkpoint(checkpoint_path)
    if loaded:
        hist_en = {"train_loss": [], "val_loss": [], "val_f1": []}
        print("  Skipping training (checkpoint loaded)", flush=True)
    else:
        en_cw = compute_weights(splits["en"]["train"]["label"].tolist())
        model_en, hist_en = train_model(
            en_train_ds, en_val_ds, class_weights=en_cw,
            save_path=checkpoint_path)
    models["mono_en"] = model_en
    histories["mono_en"] = hist_en
    eval_on_lang(model_en, "Monolingual EN", "en")

    # -- 1.3.2: multilingual --

    print("\n" + "=" * 60, flush=True)
    print("Experiment 3: Multilingual DE+EN (Task 1.3.2)")
    print("=" * 60)
    checkpoint_path = os.path.join(MODEL_DIR, "multi_de_en")
    model_multi, loaded = load_model_from_checkpoint(checkpoint_path)
    if loaded:
        hist_multi = {"train_loss": [], "val_loss": [], "val_f1": []}
        print("  Skipping training (checkpoint loaded)", flush=True)
    else:
        multi_cw = compute_weights(multi_train_df["label"].tolist())
        model_multi, hist_multi = train_model(
            multi_train_ds, multi_val_ds, class_weights=multi_cw,
            save_path=checkpoint_path)
    models["multi_de_en"] = model_multi
    histories["multi_de_en"] = hist_multi
    eval_on_lang(model_multi, "Multilingual DE+EN", "de")
    eval_on_lang(model_multi, "Multilingual DE+EN", "en")
    eval_on_lang(model_multi, "Multilingual DE+EN", "ru")

    # -- 1.3.3: translation-based eval on RU --

    print("\n" + "=" * 60, flush=True)
    print("Task 1.3.3: Machine Translation–Based Evaluation on RU Test")
    print("=" * 60)

    # a) EN-only on RU
    print("\n--- 1.3.3a: English-only → RU test ---", flush=True)
    eval_on_lang(model_en, "EN-only → RU (1.3.3a)", "ru")

    # b) multilingual on RU
    print("\n--- 1.3.3b: Multilingual DE+EN → RU test ---", flush=True)
    eval_on_lang(model_multi, "Multi DE+EN → RU (1.3.3b)", "ru")

    # c) translated EN->RU model on RU
    print("\n--- 1.3.3c: Translated EN→RU model → RU test ---", flush=True)
    checkpoint_path = os.path.join(MODEL_DIR, "translated_en_ru")
    model_trans, loaded = load_model_from_checkpoint(checkpoint_path)
    if loaded:
        hist_trans = {"train_loss": [], "val_loss": [], "val_f1": []}
        print("  Skipping training (checkpoint loaded)", flush=True)
    else:
        trans_cw = compute_weights(translated_df["label"].tolist())
        model_trans, hist_trans = train_model(
            translated_ds, en_val_ds, class_weights=trans_cw,
            save_path=checkpoint_path)
    models["translated_en_ru"] = model_trans
    histories["translated_en_ru"] = hist_trans
    eval_on_lang(model_trans, "Translated EN→RU (1.3.3c)", "ru")

    return results, models, histories


# --- 1.4: Evaluation + error analysis ---

def compile_results_table(results):
    df = pd.DataFrame(results)
    cols = ["model", "test_lang", "macro_f1", "binary_f1", "precision", "recall", "accuracy", "auc_roc"]
    df = df[[c for c in cols if c in df.columns]]
    return df


def plot_results(results_df):
    # F1 bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    pivot = results_df.pivot_table(index="test_lang", columns="model", values="macro_f1")
    pivot.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_title("Macro F1 Score: All Models x Test Languages")
    ax.set_ylabel("Macro F1")
    ax.set_xlabel("Test Language")
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "f1_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: f1_comparison.png", flush=True)

    # heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    heatmap_data = results_df.pivot_table(index="model", columns="test_lang", values="macro_f1")
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
    ax.set_title("Macro F1 Scores: Model x Test Language")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "f1_heatmap.png"), dpi=150)
    plt.close()
    print("  Saved: f1_heatmap.png", flush=True)

    # 1.3.3 RU comparison
    task133_models = ["EN-only → RU (1.3.3a)", "Multi DE+EN → RU (1.3.3b)", "Translated EN→RU (1.3.3c)"]
    task133_df = results_df[results_df["model"].isin(task133_models)]
    if len(task133_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(task133_df["model"], task133_df["macro_f1"], color=["steelblue", "coral", "seagreen"])
        ax.set_ylabel("Macro F1")
        ax.set_title("Task 1.3.3: Machine Translation–Based Evaluation on RU Test")
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, task133_df["macro_f1"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", fontsize=11)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "task133_ru_comparison.png"), dpi=150)
        plt.close()
        print("  Saved: task133_ru_comparison.png", flush=True)


def plot_training_curves(histories):
    # skip if everything was loaded from checkpoints
    trainable = {k: v for k, v in histories.items() if len(v["train_loss"]) > 0}
    if not trainable:
        print("  No training histories to plot (all loaded from checkpoints)", flush=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, hist in trainable.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], label=name, marker="o")
        axes[1].plot(epochs, hist["val_f1"], label=name, marker="o")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Validation Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print("  Saved: training_curves.png", flush=True)


def error_analysis(models, splits, tokenizer):
    print("\n" + "=" * 60, flush=True)
    print("TASK 1.4 – Cross-Lingual Error Analysis (RU zero-shot)")
    print("=" * 60)

    test_df = splits["ru"]["test"]
    if len(test_df) == 0:
        print("  No RU test data available!", flush=True)
        return
    test_ds = make_dataset(test_df, tokenizer)

    analysis_models = ["mono_en", "multi_de_en", "translated_en_ru"]

    for mname in analysis_models:
        if mname not in models:
            continue
        model = models[mname]
        metrics, preds, labels = evaluate_model(model, test_ds, return_preds=True)
        test_copy = test_df.copy()
        test_copy["prediction"] = preds

        fn = test_copy[(test_copy["label"] == 1) & (test_copy["prediction"] == 0)]
        fp = test_copy[(test_copy["label"] == 0) & (test_copy["prediction"] == 1)]
        tp = test_copy[(test_copy["label"] == 1) & (test_copy["prediction"] == 1)]
        tn = test_copy[(test_copy["label"] == 0) & (test_copy["prediction"] == 0)]

        print(f"\n    Model: {mname}")
        print(f"      TP={len(tp)}  FP={len(fp)}  FN={len(fn)}  TN={len(tn)}")
        print(f"      Macro-F1={metrics['macro_f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")

        if len(fn) > 0:
            print(f"      Sample False Negatives (missed ADEs):")
            for _, row in fn.head(3).iterrows():
                print(f"        + {str(row['text'])[:200]}")
        if len(fp) > 0:
            print(f"      Sample False Positives:")
            for _, row in fp.head(3).iterrows():
                print(f"        - {str(row['text'])[:200]}")

    # confusion matrices
    plot_models = [m for m in analysis_models if m in models]
    fig, axes = plt.subplots(1, len(plot_models), figsize=(6 * len(plot_models), 5))
    if len(plot_models) == 1:
        axes = [axes]
    for idx, mname in enumerate(plot_models):
        model = models[mname]
        _, preds, labels = evaluate_model(model, test_ds, return_preds=True)
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                    xticklabels=["No ADE", "ADE"], yticklabels=["No ADE", "ADE"])
        axes[idx].set_title(f"{mname}\n(RU zero-shot)")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrices_ru.png"), dpi=150)
    plt.close()
    print(f"  Saved: confusion_matrices_ru.png", flush=True)


# --- Main ---

def main():
    print("=" * 60, flush=True)
    print("Task 1: Multilingual Transfer and Translation")
    print("  in Medical Social Media NLP (ADE Detection)")
    print("=" * 60)
    print(f"  Training languages: {', '.join(l.upper() for l in TRAIN_LANGS)}")
    print(f"  Zero-shot language: {ZEROSHOT_LANG.upper()}")
    print(f"  Translation: EN -> {ZEROSHOT_LANG.upper()} (3rd language)")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Device: {DEVICE}", flush=True)

    print("\n>>> Task 1.1: Loading and preparing data...", flush=True)
    train_df, dev_df = load_raw_data()
    splits = prepare_splits(train_df, dev_df)

    print("  Translating English subset to Russian...", flush=True)
    translated_df = translate_english_to_russian(splits["en"]["train"])
    report_split_sizes(splits, translated_df)

    print("\n>>> Task 1.2: Exploratory analysis...", flush=True)
    exploratory_analysis(splits, translated_df)

    print("\n>>> Task 1.3: Training models...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    results, models, histories = run_all_experiments(splits, translated_df, tokenizer)

    print("\n>>> Task 1.4: Evaluation and error analysis...", flush=True)
    results_df = compile_results_table(results)
    print("\n" + "=" * 60, flush=True)
    print("COMPLETE RESULTS TABLE")
    print("=" * 60)
    print(results_df.to_string(index=False), flush=True)
    results_df.to_csv(os.path.join(RESULT_DIR, "task1_results.csv"), index=False)

    plot_results(results_df)
    plot_training_curves(histories)
    error_analysis(models, splits, tokenizer)

    print("\n>>> Task 1 complete. All outputs saved to outputs/", flush=True)


if __name__ == "__main__":
    main()
