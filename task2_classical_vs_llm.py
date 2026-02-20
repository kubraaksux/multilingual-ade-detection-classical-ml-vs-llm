#!/usr/bin/env python3
"""
Task 2 – Classical ML vs LLMs on Iris dataset
RF baseline, DistilBERT on textified data, Flan-T5 few-shot + hybrid
"""

import os
import sys
import json
import time
import warnings
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
)
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
RESULT_DIR = os.path.join(BASE_DIR, "outputs", "results")

for d in [FIG_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

SEED = 42
TRANSFORMER_MODEL = "distilbert-base-uncased"
LLM_MODEL = "google/flan-t5-large"
TRANSFORMER_EPOCHS = 20
TRANSFORMER_BATCH = 8
TRANSFORMER_LR = 3e-5
TRANSFORMER_MAX_LEN = 64

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
print(f"Using device: {DEVICE}")


# --- Data ---

def load_and_split_iris():
    """60/20/20 stratified split."""
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = list(iris.feature_names)
    target_names = list(iris.target_names)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp
    )
    # 90/30/30

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Features: {feature_names}")
    print(f"Classes: {target_names}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, target_names


# --- 2.1: Random Forest ---

def task_2_1_random_forest(X_train, X_val, X_test, y_train, y_val, y_test,
                            feature_names, target_names):
    print("\n" + "=" * 60)
    print("TASK 2.1 – Random Forest Baseline")
    print("=" * 60)

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=SEED, class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    results = {}
    for name, X_eval, y_eval in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
        y_pred = rf.predict(X_eval)
        acc = accuracy_score(y_eval, y_pred)
        f1 = f1_score(y_eval, y_pred, average="macro")
        print(f"\n  {name} Results (Acc={acc:.4f}, Macro-F1={f1:.4f}):")
        print(classification_report(y_eval, y_pred, target_names=target_names))
        if name == "Test":
            results["accuracy"] = acc
            results["macro_f1"] = f1

    # feature importance
    importances = rf.feature_importances_
    print("  Feature Importance:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"    {name}: {imp:.4f}")

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_idx = np.argsort(importances)
    ax.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx], color="steelblue")
    ax.set_title("Random Forest Feature Importance (Iris)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "rf_feature_importance.png"), dpi=150)
    plt.close()
    print("  Saved: rf_feature_importance.png")

    # decision rules from first tree
    rules = export_text(rf.estimators_[0], feature_names=feature_names, max_depth=3)
    print(f"\n  Decision rules (first tree, depth 3):\n{rules}")

    # save
    model_path = os.path.join(MODEL_DIR, "random_forest_iris.joblib")
    joblib.dump(rf, model_path)

    return rf, results, importances, rules


# --- 2.2: Textification + DistilBERT ---

def iris_to_natural_language(features, feature_names):
    return (
        f"This flower has a sepal length of {features[0]:.1f} centimeters, "
        f"a sepal width of {features[1]:.1f} centimeters, "
        f"a petal length of {features[2]:.1f} centimeters, "
        f"and a petal width of {features[3]:.1f} centimeters."
    )


def iris_to_structured(features, feature_names):
    parts = []
    for name, val in zip(feature_names, features):
        clean_name = name.replace(" (cm)", "").replace(" ", "_")
        parts.append(f"{clean_name}={val:.1f}")
    return " | ".join(parts)


class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def train_transformer(train_texts, train_labels, val_texts, val_labels,
                       model_name=TRANSFORMER_MODEL, epochs=TRANSFORMER_EPOCHS,
                       batch_size=TRANSFORMER_BATCH, lr=TRANSFORMER_LR,
                       max_len=TRANSFORMER_MAX_LEN, save_name=""):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_enc = tokenizer(train_texts, truncation=True, padding="max_length",
                          max_length=max_len, return_tensors="pt")
    val_enc = tokenizer(val_texts, truncation=True, padding="max_length",
                        max_length=max_len, return_tensors="pt")

    train_ds = TextClassificationDataset(train_enc, train_labels)
    val_ds = TextClassificationDataset(val_enc, val_labels)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model = model.to(DEVICE)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * 0.1), total_steps
    )

    best_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += outputs.loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro")
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(train_loader):.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)

    return model, tokenizer


def evaluate_transformer(model, tokenizer, texts, labels, max_len=TRANSFORMER_MAX_LEN):
    enc = tokenizer(texts, truncation=True, padding="max_length",
                    max_length=max_len, return_tensors="pt")
    ds = TextClassificationDataset(enc, labels)
    loader = DataLoader(ds, batch_size=TRANSFORMER_BATCH)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())

    return all_preds


def task_2_2_textification(X_train, X_val, X_test, y_train, y_val, y_test,
                            feature_names, target_names):
    print("\n" + "=" * 60)
    print("TASK 2.2 – Textification and Transformer-Based Modeling")
    print("=" * 60)

    results = {}

    for repr_name, text_fn in [
        ("Natural Language", iris_to_natural_language),
        ("Structured", iris_to_structured),
    ]:
        print(f"\n  --- Representation: {repr_name} ---")

        # textify
        train_texts = [text_fn(x, feature_names) for x in X_train]
        val_texts = [text_fn(x, feature_names) for x in X_val]
        test_texts = [text_fn(x, feature_names) for x in X_test]

        # example
        print(f"    Example: {train_texts[0]}")

        print(f"    Training DistilBERT on {repr_name} representation...")
        model, tokenizer = train_transformer(
            train_texts, y_train.tolist(),
            val_texts, y_val.tolist(),
            save_name=repr_name.lower().replace(" ", "_"),
        )

        # eval
        test_preds = evaluate_transformer(model, tokenizer, test_texts, y_test.tolist())
        acc = accuracy_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds, average="macro")
        print(f"\n    Test Results (Acc={acc:.4f}, Macro-F1={f1:.4f}):")
        print(classification_report(y_test, test_preds, target_names=target_names))

        results[repr_name] = {"accuracy": acc, "macro_f1": f1, "preds": test_preds}

        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return results


# --- 2.3: LLM few-shot + hybrid ---

def select_few_shot_examples(X_train, y_train, target_names, n_per_class=2):
    """Pick samples closest to each class centroid."""
    examples = []
    for cls in range(len(target_names)):
        cls_mask = y_train == cls
        cls_data = X_train[cls_mask]
        centroid = cls_data.mean(axis=0)
        distances = np.linalg.norm(cls_data - centroid, axis=1)
        closest_idx = np.argsort(distances)[:n_per_class]
        for idx in closest_idx:
            examples.append((cls_data[idx], cls))
    return examples


def create_basic_prompt(test_sample, few_shot_examples, feature_names, target_names):
    prompt = "Classify the iris flower. Classes: setosa, versicolor, virginica.\n\n"
    for i, (features, label) in enumerate(few_shot_examples):
        desc = iris_to_natural_language(features, feature_names)
        prompt += f"Example: {desc}\nClass: {target_names[label]}\n\n"

    test_desc = iris_to_natural_language(test_sample, feature_names)
    prompt += f"Classify: {test_desc}\nAnswer with one word (setosa, versicolor, or virginica):"
    return prompt


def create_augmented_prompt(test_sample, few_shot_examples, feature_names,
                             target_names, rf_model, importances, rules):
    """Prompt with RF prediction + decision rules baked in."""
    rf_pred = rf_model.predict([test_sample])[0]
    rf_proba = rf_model.predict_proba([test_sample])[0]
    prompt = "Classify the iris flower. Classes: setosa, versicolor, virginica.\n"
    prompt += "Rules: petal_length<2.5->setosa; petal_width<1.75->versicolor; else->virginica.\n"
    prompt += f"ML model predicts: {target_names[rf_pred]} ({max(rf_proba)*100:.0f}% confident).\n"

    # 1 example per class (keep it short for T5)
    for cls in range(len(target_names)):
        for feat, lbl in few_shot_examples:
            if lbl == cls:
                prompt += (
                    f"Example: sepal={feat[0]:.1f},{feat[1]:.1f} "
                    f"petal={feat[2]:.1f},{feat[3]:.1f} -> {target_names[lbl]}\n"
                )
                break

    prompt += (
        f"Flower: sepal={test_sample[0]:.1f},{test_sample[1]:.1f} "
        f"petal={test_sample[2]:.1f},{test_sample[3]:.1f}\n"
        f"Answer with one word (setosa, versicolor, or virginica):"
    )
    return prompt


def parse_llm_output(output, target_names):
    output_lower = output.strip().lower()
    for i, name in enumerate(target_names):
        if name.lower() in output_lower:
            return i
    return -1


def task_2_3_llm_fewshot(X_train, X_test, y_train, y_test,
                          feature_names, target_names, rf_model, importances, rules):
    print("\n" + "=" * 60)
    print("TASK 2.3 – LLM-Based Prediction and Hybrid Modeling")
    print("=" * 60)

    print("  Loading Flan-T5-large model...")
    tokenizer = T5Tokenizer.from_pretrained(LLM_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(LLM_MODEL)
    model = model.to(DEVICE)
    model.eval()

    few_shot_examples = select_few_shot_examples(X_train, y_train, target_names, n_per_class=2)

    results = {}
    for setup_name, prompt_fn in [
        ("LLM Few-shot", create_basic_prompt),
        ("LLM Augmented", None),  # special handling
    ]:
        print(f"\n  --- {setup_name} ---")
        preds = []
        raw_outputs = []

        for i, (features, label) in enumerate(zip(X_test, y_test)):
            if setup_name == "LLM Few-shot":
                prompt = prompt_fn(features, few_shot_examples, feature_names, target_names)
            else:
                prompt = create_augmented_prompt(
                    features, few_shot_examples, feature_names,
                    target_names, rf_model, importances, rules,
                )

            inputs = tokenizer(prompt, return_tensors="pt", max_length=512,
                               truncation=True).to(DEVICE)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            raw_outputs.append(response)
            pred = parse_llm_output(response, target_names)
            preds.append(pred)

        # handle unparsed
        valid_mask = [p >= 0 for p in preds]
        valid_preds = [p for p, v in zip(preds, valid_mask) if v]
        valid_labels = [l for l, v in zip(y_test, valid_mask) if v]
        unparsed = sum(1 for p in preds if p < 0)

        if unparsed > 0:
            print(f"    Warning: {unparsed}/{len(preds)} outputs could not be parsed")
            print(f"    Unparsed examples: {[r for r, p in zip(raw_outputs, preds) if p < 0][:5]}")

        if valid_preds:
            acc = accuracy_score(valid_labels, valid_preds)
            f1 = f1_score(valid_labels, valid_preds, average="macro")
            print(f"    Test Results (Acc={acc:.4f}, Macro-F1={f1:.4f}):")
            print(classification_report(
                valid_labels, valid_preds, target_names=target_names,
                zero_division=0,
            ))
            results[setup_name] = {
                "accuracy": acc,
                "macro_f1": f1,
                "preds": preds,
                "raw_outputs": raw_outputs,
            }
        else:
            print("    No valid predictions could be parsed.")
            results[setup_name] = {"accuracy": 0, "macro_f1": 0, "preds": preds, "raw_outputs": raw_outputs}

        # samples
        print(f"    Sample outputs:")
        for j in range(min(3, len(raw_outputs))):
            true_label = target_names[y_test[j]]
            print(f"      True: {true_label} | LLM output: '{raw_outputs[j]}' | Parsed: {preds[j]}")

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return results


# --- 2.4: Comparison ---

def task_2_4_comparison(rf_results, transformer_results, llm_results,
                         y_test, target_names):
    print("\n" + "=" * 60)
    print("TASK 2.4 – Comparative Analysis and Reflection")
    print("=" * 60)

    # results table
    rows = [
        {"Approach": "Random Forest", **rf_results},
        {"Approach": "DistilBERT (Natural Lang.)",
         "accuracy": transformer_results["Natural Language"]["accuracy"],
         "macro_f1": transformer_results["Natural Language"]["macro_f1"]},
        {"Approach": "DistilBERT (Structured)",
         "accuracy": transformer_results["Structured"]["accuracy"],
         "macro_f1": transformer_results["Structured"]["macro_f1"]},
    ]
    for setup in ["LLM Few-shot", "LLM Augmented"]:
        if setup in llm_results:
            rows.append({
                "Approach": setup,
                "accuracy": llm_results[setup]["accuracy"],
                "macro_f1": llm_results[setup]["macro_f1"],
            })

    results_df = pd.DataFrame(rows)
    print("\n  COMPLETE RESULTS TABLE:")
    print(results_df[["Approach", "accuracy", "macro_f1"]].to_string(index=False))
    results_df.to_csv(os.path.join(RESULT_DIR, "task2_results.csv"), index=False)

    # bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.35
    ax.bar(x - width / 2, results_df["accuracy"], width, label="Accuracy", color="steelblue")
    ax.bar(x + width / 2, results_df["macro_f1"], width, label="Macro F1", color="coral")
    ax.set_ylabel("Score")
    ax.set_title("Comparison of All Approaches on Iris Test Set")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Approach"], rotation=20, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    for i, (a, f) in enumerate(zip(results_df["accuracy"], results_df["macro_f1"])):
        ax.text(i - width / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, f + 0.02, f"{f:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task2_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: task2_comparison.png")

    # confusion matrices

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    approach_names = results_df["Approach"].tolist()

    pred_dict = {}
    if "Natural Language" in transformer_results:
        pred_dict["DistilBERT (Natural Lang.)"] = transformer_results["Natural Language"]["preds"]
    if "Structured" in transformer_results:
        pred_dict["DistilBERT (Structured)"] = transformer_results["Structured"]["preds"]
    for setup in ["LLM Few-shot", "LLM Augmented"]:
        if setup in llm_results and llm_results[setup]["preds"]:
            preds = llm_results[setup]["preds"]
            preds = [p if p >= 0 else 0 for p in preds]
            pred_dict[setup] = preds

    idx = 0
    for name in approach_names:
        if name in pred_dict and pred_dict[name] is not None:
            cm = confusion_matrix(y_test, pred_dict[name])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                        xticklabels=target_names, yticklabels=target_names)
            axes[idx].set_title(name)
            axes[idx].set_xlabel("Predicted")
            axes[idx].set_ylabel("True")
        idx += 1

    for i in range(idx, len(axes)):
        axes[i].set_visible(False)
    plt.suptitle("Confusion Matrices – All Approaches", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "task2_confusion_matrices.png"), dpi=150)
    plt.close()
    print("  Saved: task2_confusion_matrices.png")

    print("\n  DISCUSSION:")
    print("  RF works best on Iris -- small, clean, tabular data.")
    print("  Textification loses numerical precision; LLMs struggle")
    print("  with exact feature comparisons. For tabular data,")
    print("  classical ML is the practical choice.")

    return results_df


# --- Main ---

def main():
    print("=" * 60)
    print("Task 2: Classical ML vs LLMs on Structured Data")
    print("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, target_names = \
        load_and_split_iris()

    rf_model, rf_results, importances, rules = task_2_1_random_forest(
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names, target_names
    )

    transformer_results = task_2_2_textification(
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names, target_names
    )

    llm_results = task_2_3_llm_fewshot(
        X_train, X_test, y_train, y_test,
        feature_names, target_names, rf_model, importances, rules
    )

    comparison_df = task_2_4_comparison(
        rf_results, transformer_results, llm_results, y_test, target_names
    )

    print("\n>>> Task 2 complete. All outputs saved to outputs/")


if __name__ == "__main__":
    main()
