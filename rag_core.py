"""
rag_core.py — An Intelligent Academic Assistant for Document-Based Question Answering
IE624 project: Retrieval-Augmented Generation (RAG) pipeline.

Pipeline (matches Section 4 of the pre-project report):

  OFFLINE INDEXING                         ONLINE QUESTION ANSWERING
  A. PDF collection                        F. User question
  B. Text extraction (page metadata)       G. Query embedding + top-k cosine search
  C. Cleaning + overlapping chunking       H. Evidence-first prompt construction
  D. Embedding generation                  I. Answer generation (pretrained model)
  E. Index storage (vectors + metadata)    J. Answer + citations from stored metadata

Single-document mode: one PDF is uploaded and all questions are answered from it
(set_single_pdf). After indexing, suggested questions about the PDF are generated
(suggest_questions), NotebookLM-style.

Libraries: PyMuPDF (fitz), sentence-transformers, NumPy, Hugging Face Transformers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------------------
# Configuration (all experiment settings in one place, for reproducibility — §5.7)
# --------------------------------------------------------------------------------------


@dataclass
class RAGConfig:
    # Embedding model (§5.3)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Chunking (§5.3): ~300 words with 50 words overlap
    chunk_size_words: int = 300
    chunk_overlap_words: int = 50
    min_chunk_words: int = 40          # trailing fragments shorter than this are merged
    # Retrieval (§5.4)
    top_k: int = 3
    # Abstention gate: if the best cosine similarity is below this, the assistant says the
    # documents do not contain enough information WITHOUT calling the generator.
    # 0.30 is a starting value for all-MiniLM-L6-v2; tune it on the evaluation set.
    min_similarity: float = 0.30
    # Generator (§5.5). Default is a text-to-text model, as in the report.
    # Alternative (stronger, still free): "Qwen/Qwen2.5-1.5B-Instruct" (decoder/chat model).
    generator_model: str = "google/flan-t5-base"
    max_input_tokens: int = 1024       # question + context are truncated to this
    max_new_tokens: int = 200
    # Decoding: greedy (deterministic) so baseline and RAG are compared fairly (§6.2)
    num_beams: int = 1
    do_sample: bool = False
    repetition_penalty: float = 1.15
    # Suggested questions shown after the PDF is uploaded (NotebookLM-style)
    n_suggestions: int = 8


INSUFFICIENT_EVIDENCE = (
    "The available documents do not contain enough information to answer this question."
)

# --------------------------------------------------------------------------------------
# B. Text extraction
# --------------------------------------------------------------------------------------


def extract_pages(pdf_path: str | Path) -> List[Dict]:
    """Extract text page by page, keeping filename and 1-based page number.

    Uses PyMuPDF (fitz) as planned in §5.1; falls back to pypdf if PyMuPDF is missing.
    """
    pdf_path = Path(pdf_path)
    pages: List[Dict] = []
    try:
        import fitz  # PyMuPDF

        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                # sort=True orders text blocks top-to-bottom, left-to-right
                pages.append({"source": pdf_path.name, "page": i + 1,
                              "text": page.get_text("text", sort=True)})
    except ImportError:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            pages.append({"source": pdf_path.name, "page": i + 1,
                          "text": page.extract_text() or ""})
    return pages


# --------------------------------------------------------------------------------------
# C. Cleaning and chunking
# --------------------------------------------------------------------------------------

_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi",
              "ﬄ": "ffl", "­": "", "​": ""}


def clean_text(text: str) -> str:
    """Conservative cleaning (§5.2): fixes extraction noise, keeps punctuation/equations."""
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    # Re-join words hyphenated across a line break: "classi-\nfication" -> "classification"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    lines = []
    for line in text.splitlines():
        s = line.strip()
        # Drop lines that are only a page number (e.g. "12", "- 12 -", "Page 12")
        if re.fullmatch(r"(page\s*)?[-–]?\s*\d{1,4}\s*[-–]?", s, flags=re.I):
            continue
        lines.append(s)
    text = " ".join(lines)
    return re.sub(r"\s+", " ", text).strip()


def is_low_quality(text: str) -> bool:
    """Flag pages that are probably scanned/unreadable (little or garbled text)."""
    if len(text) < 30:
        return True
    letters = sum(c.isalpha() for c in text)
    return letters / max(len(text), 1) < 0.4


@dataclass
class Chunk:
    chunk_id: int
    source: str
    page_start: int
    page_end: int
    text: str
    n_words: int

    @property
    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}–{self.page_end}"


def chunk_document(pages: List[Dict], size: int, overlap: int, min_words: int,
                   start_id: int = 0) -> List[Chunk]:
    """Split one document into overlapping word windows.

    Words are tagged with their page number, so a chunk that crosses a page boundary
    keeps an exact page range (e.g. pp. 4–5) — citations come from metadata, never from
    the model (§4.3).
    """
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    words, word_pages = [], []
    for p in pages:
        w = clean_text(p["text"]).split()
        words.extend(w)
        word_pages.extend([p["page"]] * len(w))
    if not words:
        return []

    source = pages[0]["source"]
    step = size - overlap
    starts = list(range(0, max(len(words) - overlap, 1), step))
    chunks: List[Chunk] = []
    for n, s in enumerate(starts):
        e = min(s + size, len(words))
        # Merge a short trailing fragment into the previous chunk instead of indexing it alone
        if n > 0 and (e - s) < min_words and chunks:
            prev = chunks[-1]
            extra = words[prev_end:e]
            if extra:
                prev.text += " " + " ".join(extra)
                prev.n_words += len(extra)
                prev.page_end = word_pages[e - 1]
            break
        chunks.append(Chunk(chunk_id=start_id + len(chunks), source=source,
                            page_start=word_pages[s], page_end=word_pages[e - 1],
                            text=" ".join(words[s:e]), n_words=e - s))
        prev_end = e
        if e == len(words):
            break
    return chunks


# --------------------------------------------------------------------------------------
# D + E. Embeddings and index storage
# --------------------------------------------------------------------------------------

_MODEL_CACHE: Dict[str, object] = {}


def get_embedder(model_name: str):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def embed_texts(texts: List[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    """Return L2-normalised embeddings, so cosine similarity = dot product."""
    model = get_embedder(model_name)
    vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=len(texts) > 64,
                        convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype(np.float32)


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class DocumentIndex:
    """Vectors + chunk text + source metadata (stage E)."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.documents: List[Dict] = []     # per-file report (pages, skipped pages, hash)
        self.folder: Optional[Path] = None  # where the index is saved (suggestions cache)

    # ---- building -------------------------------------------------------------------
    def build(self, pdf_paths: List[str | Path], verbose: bool = True) -> "DocumentIndex":
        cfg = self.config
        self.chunks, self.documents = [], []
        for path in sorted(Path(p) for p in pdf_paths):
            pages = extract_pages(path)
            good = [p for p in pages if not is_low_quality(clean_text(p["text"]))]
            skipped = [p["page"] for p in pages if p not in good]
            doc_chunks = chunk_document(good, cfg.chunk_size_words, cfg.chunk_overlap_words,
                                        cfg.min_chunk_words, start_id=len(self.chunks))
            self.chunks.extend(doc_chunks)
            self.documents.append({"source": path.name, "pages": len(pages),
                                   "skipped_pages": skipped, "chunks": len(doc_chunks),
                                   "md5": _file_hash(path)})
            if verbose:
                note = f" (skipped low-text pages: {skipped})" if skipped else ""
                print(f"  {path.name}: {len(pages)} pages -> {len(doc_chunks)} chunks{note}")
        if not self.chunks:
            raise ValueError("No extractable text found in the supplied PDFs.")
        t0 = time.time()
        self.embeddings = embed_texts([c.text for c in self.chunks], cfg.embedding_model)
        if verbose:
            print(f"Embedded {len(self.chunks)} chunks with {cfg.embedding_model} "
                  f"in {time.time() - t0:.1f}s -> matrix {self.embeddings.shape}")
        return self

    # ---- persistence ----------------------------------------------------------------
    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.folder = folder
        stale = folder / "suggestions.json"      # suggestions belong to the old index
        if stale.exists():
            stale.unlink()
        np.save(folder / "embeddings.npy", self.embeddings)
        with open(folder / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False, indent=1)
        with open(folder / "index_info.json", "w", encoding="utf-8") as f:
            json.dump({"config": asdict(self.config), "documents": self.documents}, f, indent=2)

    @classmethod
    def load(cls, folder: str | Path, config: Optional[RAGConfig] = None) -> "DocumentIndex":
        folder = Path(folder)
        info = json.loads((folder / "index_info.json").read_text(encoding="utf-8"))
        saved = RAGConfig(**info["config"])
        if config is None:
            config = saved
        elif config.embedding_model != saved.embedding_model:
            raise ValueError("Index was built with a different embedding model; rebuild it.")
        idx = cls(config)
        idx.embeddings = np.load(folder / "embeddings.npy")
        idx.chunks = [Chunk(**c) for c in
                      json.loads((folder / "chunks.json").read_text(encoding="utf-8"))]
        idx.documents = info["documents"]
        idx.folder = folder
        return idx

    # ---- G. retrieval ---------------------------------------------------------------
    def search(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """Embed the question and rank all chunks by cosine similarity; return top-k."""
        k = k or self.config.top_k
        q = embed_texts([query], self.config.embedding_model)[0]
        scores = self.embeddings @ q                      # cosine (vectors are normalised)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [{"rank": r + 1, "score": float(scores[i]), "chunk": self.chunks[i]}
                for r, i in enumerate(top)]


# --------------------------------------------------------------------------------------
# H. Prompt construction (§5.5)
# --------------------------------------------------------------------------------------


def build_rag_prompt(question: str, hits: List[Dict]) -> str:
    context = "\n\n".join(
        f"[{h['rank']}] ({h['chunk'].source}, {h['chunk'].page_label}) {h['chunk'].text}"
        for h in hits)
    # Question is placed BEFORE the context so that, if the input is truncated,
    # only the tail of the context is lost — never the question or the instructions.
    return (
        "Role: You are an academic study assistant.\n"
        "Task: Answer the user's question using only the provided context.\n"
        "Limitations: If the context does not support an answer, reply exactly: "
        f"\"{INSUFFICIENT_EVIDENCE}\"\n"
        "Style: Give a concise, clear explanation in complete sentences. "
        "Do not invent citations.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )


def build_baseline_prompt(question: str) -> str:
    """No-retrieval baseline (§6.2): same model, same decoding, no document context."""
    return (
        "Role: You are an academic study assistant.\n"
        "Task: Answer the user's question. If you do not know the answer, reply exactly: "
        f"\"{INSUFFICIENT_EVIDENCE}\"\n"
        "Style: Give a concise, clear explanation in complete sentences.\n\n"
        f"Question: {question}\n\nAnswer:"
    )


# --------------------------------------------------------------------------------------
# I. Answer generation
# --------------------------------------------------------------------------------------


class Generator:
    """Wraps either a text-to-text model (FLAN-T5) or a small chat model (e.g. Qwen2.5)."""

    def __init__(self, config: RAGConfig):
        import torch
        from transformers import (AutoConfig, AutoModelForCausalLM,
                                  AutoModelForSeq2SeqLM, AutoTokenizer)

        self.config = config
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        name = config.generator_model
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.is_seq2seq = AutoConfig.from_pretrained(name).is_encoder_decoder
        dtype = torch.float16 if self.device == "cuda" and not self.is_seq2seq else torch.float32
        loader = AutoModelForSeq2SeqLM if self.is_seq2seq else AutoModelForCausalLM
        self.model = loader.from_pretrained(name, torch_dtype=dtype).to(self.device).eval()

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        cfg, tok, torch = self.config, self.tokenizer, self.torch
        if not self.is_seq2seq and getattr(tok, "chat_template", None):
            prompt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                             tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt", truncation=True,
                     max_length=cfg.max_input_tokens).to(self.device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens or cfg.max_new_tokens,
                          num_beams=cfg.num_beams,
                          do_sample=cfg.do_sample, repetition_penalty=cfg.repetition_penalty)
        if tok.pad_token_id is None:
            gen_kwargs["pad_token_id"] = tok.eos_token_id
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        if not self.is_seq2seq:                     # strip the echoed prompt
            out = out[:, inputs["input_ids"].shape[1]:]
        return tok.decode(out[0], skip_special_tokens=True).strip()


_ABSTAIN_PATTERNS = [
    "do not contain enough information", "does not contain enough information",
    "not enough information", "cannot be answered", "can't be answered",
    "not mentioned in the context", "context does not", "no information",
    "i don't know", "i do not know", "unanswerable",
]


def is_abstention(answer: str) -> bool:
    a = answer.lower()
    return any(p in a for p in _ABSTAIN_PATTERNS)


# --------------------------------------------------------------------------------------
# Assistant: F → J end to end
# --------------------------------------------------------------------------------------


@dataclass
class Answer:
    question: str
    answer: str
    abstained: bool
    sources: List[Dict] = field(default_factory=list)   # citations from stored metadata
    best_score: float = 0.0
    seconds: float = 0.0
    mode: str = "rag"


class AcademicAssistant:
    def __init__(self, index: Optional[DocumentIndex], config: RAGConfig,
                 generator: Optional[Generator] = None):
        self.index = index
        self.config = config
        self._generator = generator

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator(self.config)
        return self._generator

    def ask(self, question: str, k: Optional[int] = None,
            min_similarity: Optional[float] = None) -> Answer:
        """RAG answer with citations. Response time covers retrieval + generation."""
        t0 = time.time()
        k = k or self.config.top_k
        thr = self.config.min_similarity if min_similarity is None else min_similarity
        hits = self.index.search(question, k)
        best = hits[0]["score"] if hits else 0.0
        sources = [{"rank": h["rank"], "score": round(h["score"], 3),
                    "source": h["chunk"].source, "pages": h["chunk"].page_label,
                    "page_start": h["chunk"].page_start, "page_end": h["chunk"].page_end,
                    "text": h["chunk"].text} for h in hits]

        if best < thr:      # evidence gate: nothing relevant retrieved → abstain
            return Answer(question, INSUFFICIENT_EVIDENCE, True, sources, best,
                          time.time() - t0)

        text = self.generator.generate(build_rag_prompt(question, hits))
        abstained = is_abstention(text) or not text
        if abstained:
            text = INSUFFICIENT_EVIDENCE
        return Answer(question, text, abstained, sources, best, time.time() - t0)

    def ask_baseline(self, question: str) -> Answer:
        t0 = time.time()
        text = self.generator.generate(build_baseline_prompt(question))
        return Answer(question, text, is_abstention(text), [], 0.0, time.time() - t0,
                      mode="baseline")

    # ---- suggested questions (NotebookLM-style) ---------------------------------------
    def suggested_questions(self, n: Optional[int] = None, refresh: bool = False,
                            seed: int = 0) -> List[str]:
        """Questions generated from the uploaded documents, cached in the index folder."""
        n = n or self.config.n_suggestions
        cache = self.index.folder / "suggestions.json" if self.index.folder else None
        if cache and cache.exists() and not refresh:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if len(cached) >= n:
                self._shown_suggestions = cached[:n]
                return cached[:n]
        shown = getattr(self, "_shown_suggestions", []) if refresh else []
        qs = suggest_questions(self.index, self.generator, n=n, seed=seed,
                               min_similarity=self.config.min_similarity, exclude=shown)
        if not qs:                         # nothing new left in a short PDF → reuse list
            qs = shown[-n:] if shown else suggest_questions(
                self.index, self.generator, n=n, min_similarity=self.config.min_similarity)
        self._shown_suggestions = shown + [q for q in qs if q not in shown]
        if cache:
            cache.write_text(json.dumps(qs, ensure_ascii=False, indent=1), encoding="utf-8")
        return qs



# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# Suggested questions
# --------------------------------------------------------------------------------------

_NOISE_PATTERNS = re.compile(
    r"(et al\.|doi|https?://|www\.|\.{4,}|internet source|student paper|similarity index|"
    r"submission id|<1%|isbn|vol\.|pp\.\s*\d)", re.I)


def is_contentful(text: str) -> bool:
    """Skip chunks that make poor question sources: references, table of contents,
    plagiarism-report pages, tables of numbers."""
    words = text.split()
    if len(words) < 80:
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters / max(len(text), 1) < 0.65:
        return False
    return len(_NOISE_PATTERNS.findall(text)) / len(words) < 0.02


def pick_representative_chunks(index: DocumentIndex, n: int, seed: int = 0,
                               diversity: float = 0.6) -> List[Chunk]:
    """Choose chunks that are typical of their document but different from each other
    (maximal marginal relevance), spread across documents in turn.
    seed=0 gives the first n in that order, seed=1 the next n, and so on (wrapping)."""
    order = _mmr_order(index, n * (seed + 1), diversity)
    if not order:
        return []
    start = (n * seed) % len(order)
    return (order[start:] + order[:start])[:n]


def _mmr_order(index: DocumentIndex, n: int, diversity: float) -> List[Chunk]:
    cand = [c for c in index.chunks if is_contentful(c.text)] or list(index.chunks)
    by_doc: Dict[str, List[Chunk]] = {}
    for c in cand:
        by_doc.setdefault(c.source, []).append(c)
    E = index.embeddings
    picked: List[Chunk] = []
    pools = {}
    for doc, cs in by_doc.items():
        ids = np.array([c.chunk_id for c in cs])
        centroid = E[ids].mean(0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        rep = E[ids] @ centroid
        pools[doc] = (list(ids), list(rep))
    docs = list(pools)
    while len(picked) < n and any(pools[d][0] for d in docs):
        for d in docs:
            ids, rep = pools[d]
            if not ids or len(picked) >= n:
                continue
            if picked:
                sel = np.array([c.chunk_id for c in picked])
                redund = (E[ids] @ E[sel].T).max(1)
            else:
                redund = np.zeros(len(ids))
            score = (1 - diversity) * np.array(rep) - diversity * redund
            j = int(np.argmax(score))
            picked.append(index.chunks[ids[j]])
            ids.pop(j), rep.pop(j)
    return picked


# Question styles are rotated so the suggestion list mixes question types, like the
# evaluation set (factual / conceptual / comparison / findings).
QUESTION_STYLES = [
    "a factual question (what / which / how many) about a specific definition, number, "
    "method or result",
    "a conceptual question (why or how) about the reasoning, mechanism or purpose behind "
    "an idea",
    "a question comparing two concepts, methods, cases or results discussed in the text "
    "(if only one is discussed, ask how it works instead)",
    "a question about a finding, conclusion, assumption or limitation stated in the text",
]

QUESTION_PROMPT = (
    "Read the text below from an academic document. Write ONE clear question that a "
    "student could ask: {style}. The question must be answerable from the text, "
    "be specific (mention the concept, method or result by name), and must not refer to "
    "\"the text\" or \"the passage\". Reply with the question only.\n\n"
    "Text: {text}\n\nQuestion:"
)


def _clean_question(raw: str) -> Optional[str]:
    for line in raw.strip().splitlines():
        line = re.sub(r"^\s*(question\s*\d*\s*[:.-]|\d+[.)]|[-*•])\s*", "", line,
                      flags=re.I).strip().strip('"').strip()
        if not line:
            continue
        if "?" in line:
            line = line[: line.index("?") + 1]
        elif re.match(r"(explain|describe|discuss|summari[sz]e|compare|list|outline|define)\b",
                      line, re.I):
            line = line.rstrip(".")
        else:
            continue
        n_words = len(line.split())
        if 4 <= n_words <= 30 and not re.search(r"\b(the|this) (text|passage|excerpt)\b",
                                                 line, re.I):
            return line[0].upper() + line[1:]
    return None


def suggest_questions(index: DocumentIndex, generator: "Generator", n: int = 6,
                      seed: int = 0, min_similarity: float = 0.3,
                      exclude: Optional[List[str]] = None) -> List[str]:
    """Generate n clickable starter questions from the indexed documents.

    1. pick 2n representative, diverse, content-rich chunks;
    2. ask the generator for one question per chunk;
    3. keep a question only if retrieval for it finds its source passage in the top 3
       (or the best similarity ≥ min_similarity), and it is not a near-duplicate.
    A broad overview question is always offered first.
    """
    overview = ("What is the main objective and key contribution of this document?"
                if len(index.documents) == 1 else
                "What are the main topics covered in these documents?")
    questions: List[str] = []
    q_vecs: List[np.ndarray] = []
    if seed == 0:       # offer the overview question only if it can be answered from the PDF
        ov = embed_texts([overview], index.config.embedding_model)[0]
        if float((index.embeddings @ ov).max()) >= min_similarity:
            questions.append(overview)
    if exclude:                              # questions already shown → don't repeat them
        q_vecs = list(embed_texts(list(exclude), index.config.embedding_model))
    for attempt, chunk in enumerate(pick_representative_chunks(index, 2 * n, seed=seed)):
        if len(questions) >= n:
            break
        text = " ".join(chunk.text.split()[:350])
        style = QUESTION_STYLES[(attempt + seed) % len(QUESTION_STYLES)]
        q = _clean_question(generator.generate(QUESTION_PROMPT.format(style=style, text=text),
                                               max_new_tokens=60))
        if not q:
            continue
        v = embed_texts([q], index.config.embedding_model)[0]
        if q_vecs and max(float(v @ u) for u in q_vecs) > 0.85:      # near-duplicate
            continue
        # keep only questions the retriever can answer from this PDF: its source passage
        # must come back in the top 3, or the best match must pass the answer threshold
        scores = index.embeddings @ v
        top3 = np.argsort(-scores)[:3]
        if chunk.chunk_id not in top3 and float(scores.max()) < min_similarity:
            continue
        q_vecs.append(v)
        questions.append(q)
    fallbacks = ["What methodology or approach is used?", "What are the main findings or results?",
                 "What conclusions and limitations are discussed?",
                 "What are the key terms and definitions introduced?"]
    for f in fallbacks:
        if len(questions) >= n:
            break
        if f not in (exclude or []):
            questions.append(f)
    return questions[:n]


# --------------------------------------------------------------------------------------
# J. Presentation helpers
# --------------------------------------------------------------------------------------

def format_sources_markdown(ans: Answer, excerpt_words: int = 70) -> str:
    if not ans.sources:
        return "_No sources (baseline mode)._"
    lines = []
    if ans.abstained:
        lines.append(f"> ⚠️ Best match similarity was {ans.best_score:.2f}. The closest "
                     "passages are shown below, but they were judged insufficient.\n")
    for s in ans.sources:
        words = s["text"].split()
        excerpt = " ".join(words[:excerpt_words]) + (" …" if len(words) > excerpt_words else "")
        lines.append(f"**[{s['rank']}] {s['source']} — {s['pages']}** "
                     f"(similarity {s['score']:.2f})\n\n> {excerpt}\n")
    lines.append(f"_Response time: {ans.seconds:.2f} s_")
    return "\n".join(lines)


def format_answer_with_citations(ans: Answer) -> str:
    if ans.abstained or not ans.sources:
        return ans.answer
    cites = "; ".join(f"[{s['rank']}] {s['source']}, {s['pages']}" for s in ans.sources)
    return f"{ans.answer}\n\nSources: {cites}"


def set_single_pdf(pdf_path: str | Path, pdf_dir: str | Path = "documents",
                   index_dir: str | Path = "index") -> Path:
    """Single-document mode: empty the documents folder (and the old index), then copy in
    the one PDF to be questioned. Returns the path of the copied PDF."""
    import shutil

    src = Path(pdf_path)
    if src.suffix.lower() != ".pdf":
        raise ValueError(f"{src.name} is not a PDF file.")
    pdf_dir = Path(pdf_dir)
    shutil.rmtree(pdf_dir, ignore_errors=True)
    shutil.rmtree(index_dir, ignore_errors=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    # tidy name: "thesis.pdf.pdf" -> "thesis.pdf"
    name = src.name
    while name.lower().endswith(".pdf.pdf"):
        name = name[:-4]
    dst = pdf_dir / name
    shutil.copy(src, dst)
    return dst


def list_pdfs(folder: str | Path) -> List[Path]:
    return sorted(p for p in Path(folder).glob("**/*") if p.suffix.lower() == ".pdf")


def build_or_load_index(pdf_folder: str | Path, index_folder: str | Path,
                        config: RAGConfig, rebuild: bool = False) -> DocumentIndex:
    """Reuse a saved index if the PDFs and chunk settings are unchanged; otherwise rebuild."""
    pdfs = list_pdfs(pdf_folder)
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {pdf_folder}")
    index_folder = Path(index_folder)
    if not rebuild and (index_folder / "index_info.json").exists():
        info = json.loads((index_folder / "index_info.json").read_text())
        same_files = {d["source"]: d["md5"] for d in info["documents"]} == \
                     {p.name: _file_hash(p) for p in pdfs}
        saved = info["config"]
        same_cfg = all(saved.get(k) == getattr(config, k) for k in
                       ("embedding_model", "chunk_size_words", "chunk_overlap_words",
                        "min_chunk_words"))
        if same_files and same_cfg:
            print(f"Loaded existing index from {index_folder}")
            return DocumentIndex.load(index_folder, config)
    print(f"Building index from {len(pdfs)} PDF(s):")
    idx = DocumentIndex(config).build(pdfs)
    idx.save(index_folder)
    return idx