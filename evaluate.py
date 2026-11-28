"""
evaluate.py — Evaluation of the RAG assistant vs. a no-retrieval baseline (§6 of the report).

Workflow
  1. Fill in eval_questions.csv (≈25 questions: 8 factual, 7 conceptual, 5 comparison,
     5 unanswerable) with expected answers and gold source/page.
  2. run_evaluation(...)  -> writes results_raw.csv  (answers, retrieved sources, timings,
     automatic metrics, and EMPTY columns for manual scoring).
  3. Open results_raw.csv, fill the manual rubric columns (0/1/2) and citation_ok (0/1),
     save as results_scored.csv.
  4. summarise(...)  -> the metrics table for the final report.

Automatic metrics: Recall@k, abstention success, response time, citation match (gold
source/page among cited chunks), token-F1 vs. expected answer (auxiliary only).
Manual metrics: answer correctness (0–2), faithfulness (0–2), citation accuracy (0/1).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from pathlib import Path
from typing import List, Optional

import pandas as pd

from rag_core import AcademicAssistant

QUESTION_COLUMNS = ["id", "question", "type", "answerable", "expected_answer",
                    "source_file", "source_pages"]


# ---------------------------------------------------------------------------- helpers

def _yes(v) -> bool:
    return str(v).strip().lower() in {"yes", "y", "true", "1"}


def parse_pages(v) -> List[int]:
    """'4' -> [4];  '4;5' or '4,5' -> [4,5];  '4-6' -> [4,5,6];  blank -> []"""
    if pd.isna(v) or str(v).strip() == "":
        return []
    pages = []
    for part in re.split(r"[;,]", str(v)):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages.extend(range(int(float(a)), int(float(b)) + 1))
        elif part:
            pages.append(int(float(part)))
    return pages


def _gold_hit(src: dict, gold_file: str, gold_pages: List[int]) -> bool:
    """A retrieved chunk is relevant if it comes from the gold file and its page range
    covers at least one gold page (or any page, if no gold page was given)."""
    def stem(name: str) -> str:
        name = str(name).strip().lower()
        while name.endswith(".pdf"):
            name = name[:-4]
        return name

    gold_files = [stem(g) for g in str(gold_file).split(";") if g.strip()]
    s = stem(src["source"])
    # exact match, or one name contains the other (handles renamed copies / ".pdf.pdf")
    if not any(g == s or (g and (g in s or s in g)) for g in gold_files):
        return False
    if not gold_pages:
        return True
    return any(src["page_start"] <= p <= src["page_end"] for p in gold_pages)


def _normalise(text: str) -> List[str]:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return text.split()


def token_f1(pred: str, gold: str) -> float:
    """SQuAD-style token overlap F1 — a rough automatic proxy, not a replacement for the
    manual correctness rubric."""
    p, g = _normalise(pred), _normalise(gold)
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------- run

_ALIASES = {
    "question": ["q", "questions", "query", "question_text"],
    "type": ["category", "question_type", "qtype", "kind"],
    "answerable": ["is_answerable", "answerable_yes_no", "answerable_y_n", "can_answer"],
    "expected_answer": ["answer", "expected", "gold_answer", "reference_answer",
                        "ground_truth", "correct_answer", "reference"],
    "source_file": ["source", "file", "document", "gold_source", "source_document",
                    "pdf", "filename", "doc"],
    "source_pages": ["page", "pages", "source_page", "gold_pages", "page_number",
                     "page_no", "gold_page"],
    "id": ["no", "sno", "s_no", "number", "qid", "question_id", "sr_no"],
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    norm = {c: re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")
            for c in df.columns}
    df = df.rename(columns=norm)
    for canon, alts in _ALIASES.items():
        if canon not in df.columns:
            for a in alts:
                if a in df.columns:
                    df = df.rename(columns={a: canon})
                    break
    return df


def load_questions(path: str | Path) -> pd.DataFrame:
    """Read the question sheet. Column names are matched loosely (e.g. "Answer", "Page",
    "Source" work). Only `question` is strictly required; missing optional columns are
    filled in with a warning (Recall@k needs source_file + source_pages)."""
    # utf-8-sig removes Excel's BOM; sep=None also accepts ;-separated Excel exports
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df = _normalise_columns(df)
    if "question" not in df.columns:
        raise ValueError(f"No 'question' column found. Columns are: {list(df.columns)}. "
                         f"Expected: {QUESTION_COLUMNS}")
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))
    if "type" not in df.columns:
        df["type"] = "unspecified"
    df["type"] = df["type"].fillna("unspecified").astype(str).str.strip().str.lower()
    if "answerable" not in df.columns:
        df["answerable"] = df["type"].ne("unanswerable").map({True: "yes", False: "no"})
        print("ℹ️ No 'answerable' column: inferred from type (unanswerable → no).")
    for col in ("expected_answer", "source_file", "source_pages"):
        if col not in df.columns:
            df[col] = ""
            print(f"⚠️ No '{col}' column: filled blank."
                  + (" Recall@k cannot be computed without it." if col != "expected_answer"
                     else " Token-F1 cannot be computed without it."))
    df[["expected_answer", "source_file", "source_pages"]] = \
        df[["expected_answer", "source_file", "source_pages"]].fillna("")
    todo = df["question"].astype(str).str.strip().str.upper().str.startswith("TODO")
    if todo.any():
        print(f"Skipping {int(todo.sum())} unfilled template row(s) (question starts with TODO).")
        df = df[~todo].copy()
    df["answerable"] = df["answerable"].map(_yes)
    return df


def run_evaluation(assistant: AcademicAssistant, questions_csv: str | Path,
                   out_csv: str | Path = "results_raw.csv", k: Optional[int] = None,
                   run_baseline: bool = True) -> pd.DataFrame:
    qs = load_questions(questions_csv)
    k = k or assistant.config.top_k
    rows = []
    for _, q in qs.iterrows():
        print(f"[{q['id']}] {q['question'][:70]}")
        gold_pages = parse_pages(q["source_pages"])
        rag = assistant.ask(q["question"], k=k)

        has_gold = q["answerable"] and str(q["source_file"]).strip() != ""
        hits = [_gold_hit(s, q["source_file"], gold_pages) for s in rag.sources] \
            if has_gold else []
        row = {
            "id": q["id"], "type": q["type"], "answerable": q["answerable"],
            "question": q["question"], "expected_answer": q["expected_answer"],
            "gold_source": q["source_file"], "gold_pages": q["source_pages"],
            # --- RAG condition
            "rag_answer": rag.answer,
            "rag_abstained": rag.abstained,
            "rag_best_similarity": round(rag.best_score, 3),
            "rag_cited": " | ".join(f"{s['source']} {s['pages']}" for s in rag.sources),
            f"recall@{k}": (any(hits) if has_gold else None),
            "first_relevant_rank": (hits.index(True) + 1 if any(hits) else None),
            "rag_seconds": round(rag.seconds, 3),
            "rag_token_f1": (round(token_f1(rag.answer, str(q["expected_answer"])), 3)
                             if q["answerable"] and q["expected_answer"] else None),
            # --- manual scoring (fill in by hand)
            "rag_correctness_0to2": "", "rag_faithfulness_0to2": "", "rag_citation_ok_0or1": "",
        }
        if run_baseline:
            base = assistant.ask_baseline(q["question"])
            row.update({
                "baseline_answer": base.answer,
                "baseline_abstained": base.abstained,
                "baseline_seconds": round(base.seconds, 3),
                "baseline_token_f1": (round(token_f1(base.answer, str(q["expected_answer"])), 3)
                                      if q["answerable"] and q["expected_answer"] else None),
                "baseline_correctness_0to2": "",
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} rows to {out_csv}. Fill the *_0to2 / *_0or1 columns, then "
          "call summarise().")
    return df


# ---------------------------------------------------------------------------- summary

def _bool_mean(series) -> Optional[float]:
    if series is None:
        return None
    s = series[series.notna() & (series.astype(str).str.strip() != "")]
    return round(float(s.map(_yes).mean()), 3) if len(s) else None


def _mean(series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(s.mean()), 3) if len(s) else None


def _mrr(ans: pd.DataFrame, rcol: str) -> Optional[float]:
    if rcol not in ans.columns:
        return None
    gold = ans[ans[rcol].notna() & (ans[rcol].astype(str).str.strip() != "")]
    if not len(gold):
        return None
    rr = 1 / pd.to_numeric(gold["first_relevant_rank"], errors="coerce")
    return round(float(rr.fillna(0).mean()), 3)


def summarise(results: str | Path | pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Build the metrics table (Table 3 of the report) for both conditions."""
    df = pd.read_csv(results) if not isinstance(results, pd.DataFrame) else results.copy()
    df["answerable"] = df["answerable"].map(_yes)
    ans, unans = df[df["answerable"]], df[~df["answerable"]]
    rcol = f"recall@{k}"
    has_base = "baseline_answer" in df.columns

    def abst_rate(col):
        if col not in df.columns or not len(unans):
            return None
        return round(unans[col].map(_yes).mean(), 3)

    def false_abst_rate(col):
        if col not in df.columns or not len(ans):
            return None
        return round(ans[col].map(_yes).mean(), 3)

    metrics = [
        (f"Recall@{k} (answerable)", _bool_mean(ans[rcol]) if rcol in df else None, "—"),
        ("Mean reciprocal rank of first relevant chunk", _mrr(ans, rcol), "—"),
        ("Answer correctness, mean 0–2 (manual)", _mean(ans.get("rag_correctness_0to2")),
         _mean(ans.get("baseline_correctness_0to2")) if has_base else "—"),
        ("Faithfulness, mean 0–2 (manual)", _mean(ans.get("rag_faithfulness_0to2")), "—"),
        ("Citation accuracy (manual)", _mean(ans.get("rag_citation_ok_0or1")), "—"),
        ("Token-F1 vs expected answer (auto, auxiliary)", _mean(ans.get("rag_token_f1")),
         _mean(ans.get("baseline_token_f1")) if has_base else "—"),
        ("Abstention success (unanswerable)", abst_rate("rag_abstained"),
         abst_rate("baseline_abstained") if has_base else "—"),
        ("False abstention rate (answerable)", false_abst_rate("rag_abstained"),
         false_abst_rate("baseline_abstained") if has_base else "—"),
        ("Mean response time (s)", _mean(df["rag_seconds"]),
         _mean(df["baseline_seconds"]) if has_base else "—"),
    ]
    table = pd.DataFrame(metrics, columns=["Metric", "RAG", "Baseline (no retrieval)"])
    table = table.fillna("not scored yet")
    return table


def breakdown_by_type(results: str | Path | pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Per question-type view (factual / conceptual / comparison / unanswerable)."""
    df = pd.read_csv(results) if not isinstance(results, pd.DataFrame) else results.copy()
    df["answerable"] = df["answerable"].map(_yes)
    out = []
    for t, g in df.groupby("type"):
        out.append({
            "type": t, "n": len(g),
            f"recall@{k}": _bool_mean(g[f"recall@{k}"]) if g["answerable"].any() else "—",
            "rag_correctness": _mean(g.get("rag_correctness_0to2")),
            "baseline_correctness": _mean(g.get("baseline_correctness_0to2")),
            "rag_abstained_%": round(100 * g["rag_abstained"].map(_yes).mean(), 1),
        })
    return pd.DataFrame(out).fillna("—")


def threshold_sweep(results: str | Path | pd.DataFrame,
                    thresholds=(0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)) -> pd.DataFrame:
    """Shows how the similarity gate alone would trade off abstention on unanswerable
    questions against false abstention on answerable ones — use it to tune
    RAGConfig.min_similarity (report the chosen value)."""
    df = pd.read_csv(results) if not isinstance(results, pd.DataFrame) else results.copy()
    df["answerable"] = df["answerable"].map(_yes)
    rows = []
    for t in thresholds:
        gated = df["rag_best_similarity"] < t
        rows.append({"min_similarity": t,
                     "unanswerable_abstained": round(gated[~df["answerable"]].mean(), 3),
                     "answerable_wrongly_gated": round(gated[df["answerable"]].mean(), 3)})
    return pd.DataFrame(rows)
