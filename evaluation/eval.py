# eval.py
"""Evaluation suite for RAG system.

- Loads corpus from `corpus.txt` where passages are separated by blank lines and start with an id line.
- Indexes passages with BM25 (rank-bm25) and optionally a dense retriever (sentence‑transformers).
- Retrieves top‑k passages per question, using a hybrid BM25 + dense score.
- Measures retrieval accuracy (expected passage appears in top‑k).
- Measures generation accuracy by checking if the gold answer string is present in the selected passage.
- Handles unanswerable questions (no relevant passage).
- Outputs a CSV `results.csv` and prints a markdown table.
"""
import json
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# ---- Optional dense retriever ------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _dense_model = SentenceTransformer('all-MiniLM-L6-v2')
    _use_dense = True
except Exception:
    _use_dense = False

# ---- Configuration ----------------------------------------------------------
CORPUS_PATH = Path(__file__).parent / "corpus.txt"
QUESTIONS_PATH = Path(__file__).parent / "questions.json"
RESULTS_PATH = Path(__file__).parent / "results.csv"
TOP_K = 5

def load_corpus() -> Tuple[List[str], List[str]]:
    """Return passage texts and their IDs.
    Each passage is prefixed with its ID (e.g., "Passage 1: …") to give the dense model a stronger signal.
    """
    raw = CORPUS_PATH.read_text(encoding="utf-8").strip().split("\n\n")
    passages: List[str] = []
    ids: List[str] = []
    for block in raw:
        lines = block.strip().splitlines()
        pid = lines[0].strip() if lines[0].strip().isdigit() else str(len(ids) + 1)
        # Text without the numeric ID line
        body = " ".join(lines[1:]).strip() if lines[0].strip().isdigit() else " ".join(lines).strip()
        passages.append(f"Passage {pid}: {body}")
        ids.append(pid)
    return passages, ids

def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

def evaluate():
    passages, passage_ids = load_corpus()
    tokenized_corpus = [tokenize(p) for p in passages]
    bm25 = BM25Okapi(tokenized_corpus)

    if _use_dense:
        passage_embs = _dense_model.encode(passages, convert_to_numpy=True, normalize_embeddings=True)
    else:
        passage_embs = None

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    rows = []
    retrieval_hits = 0
    generation_hits = 0
    total = len(questions["questions"])

    # Hybrid weight: 1 = only BM25, 0 = only dense, 0.5 = equal blend
    alpha = 0.5

    for q in tqdm(questions["questions"], desc="Evaluating"):
        qid = q["id"]
        query = q["question"]
        gold_answer = q["answer"].strip().lower()
        expected_pid = q.get("passage_id")
        unanswerable = q.get("unanswerable", False)

        if _use_dense:
            query_emb = _dense_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            dense_scores = np.dot(passage_embs, query_emb)
            bm25_scores = bm25.get_scores(tokenize(query))
            combined = alpha * bm25_scores + (1 - alpha) * dense_scores
            top_indexes = combined.argsort()[::-1][:TOP_K]
        else:
            bm25_scores = bm25.get_scores(tokenize(query))
            top_indexes = bm25_scores.argsort()[::-1][:TOP_K]

        top_pids = [passage_ids[i] for i in top_indexes]
        top_texts = [passages[i] for i in top_indexes]

        # Retrieval correctness
        if expected_pid is not None:
            retrieval_correct = expected_pid in top_pids
        else:
            retrieval_correct = False
        if unanswerable:
            retrieval_correct = expected_pid not in top_pids

        # Generation correctness
        if unanswerable:
            generation_correct = all(gold_answer not in txt.lower() for txt in top_texts)
        else:
            if expected_pid and expected_pid in top_pids:
                idx = top_pids.index(expected_pid)
                candidate = top_texts[idx]
            else:
                candidate = top_texts[0]
            generation_correct = gold_answer in candidate.lower()

        retrieval_hits += retrieval_correct
        generation_hits += generation_correct

        rows.append({
            "question_id": qid,
            "question": query,
            "gold_answer": gold_answer,
            "expected_passage_id": expected_pid,
            "retrieved_passage_ids": ",".join(top_pids),
            "retrieval_correct": retrieval_correct,
            "generation_correct": generation_correct,
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_PATH, index=False)
    retrieval_acc = retrieval_hits / total * 100
    generation_acc = generation_hits / total * 100
    print("## Evaluation Summary")
    print(f"- Total questions: {total}")
    print(f"- Retrieval accuracy (top {TOP_K}): {retrieval_acc:.2f}% ({retrieval_hits}/{total})")
    print(f"- Generation accuracy: {generation_acc:.2f}% ({generation_hits}/{total})")
    print(df.head(20).to_markdown(index=False))

if __name__ == "__main__":
    evaluate()
