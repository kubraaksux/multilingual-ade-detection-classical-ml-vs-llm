# Multilingual ADE Detection and Classical ML vs. LLM Comparison

## Project Structure

```
task1_multilingual.py     # Task 1: Multilingual ADE detection (XLM-RoBERTa, DE/EN/RU)
task2_classical_vs_llm.py # Task 2: Classical ML vs LLM on Iris (RF, DistilBERT, Flan-T5)
data/raw/                 # SMM4H 2026 Task 1 dataset (not in repo)
data/processed/           # Translated datasets (EN->DE, EN->FR, EN->RU)
outputs/models/           # Trained model checkpoints (XLM-RoBERTa-base)
outputs/figures/          # Generated plots
outputs/results/          # Result CSVs
report/                   # LaTeX report + bibliography
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# Task 1 - Multilingual ADE detection (trains models if not found in outputs/models/)
python task1_multilingual.py

# Task 2 - Classical ML vs LLM comparison
python task2_classical_vs_llm.py
```

## Notes

- Task 1 models are ~1.1 GB each and are not stored in the repo. Running `task1_multilingual.py` will retrain them automatically if they are missing.
- Raw data (`data/raw/`) needs to be downloaded separately from the SMM4H 2026 shared task.
