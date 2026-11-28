# T-RAG: Financial Market Intelligence using Temporal-Aware RAG

**IE624 course project, IIT Bombay (Jul–Nov 2026)**
**Team :** K Sri Harsha(23b3951) & M Manidheep Reddy(23b1311)

## What this project is

Questions about markets are almost always questions about time: *How has Tesla's sentiment changed over the last two years? What is the latest revenue Apple reported?* Standard retrieval-augmented generation (RAG) finds text that is similar in meaning to a question, but it largely ignores dates. It mixes up years and can even use information from after the date the question is asked about.

**T-RAG** is a time-aware question-answering system for ten large US technology stocks (Tesla, Apple, Microsoft, Nvidia, Amazon, Meta, Alphabet, Netflix, AMD, Intel) over 2021–2023. It combines news, tweets, earnings press releases, reported revenue, Fed statements, government policy actions and daily prices. A router works out the time window of each question, the search only uses documents published inside that window and before the question's date, and tools give the model exact numbers (sentiment timeline, events, prices, a temporal knowledge graph) before it answers step by step with citations.

## Main results

On **TempFinQA-v2**, our test set of 317 questions with known answers:

| Version | Accuracy |
|---|---|
| B0 : LLM alone (Qwen2.5-7B-Instruct) | 18.3% |
| B2 : standard hybrid RAG, no notion of time | 47.0% |
| B6 : full T-RAG | **87.1%** |

* 14.7% of standard RAG's evidence came from **after** the question's date; T-RAG used none.
* Asked "What happened to Tesla in 2022?" without a date filter, only 68% (semantic), 72% (keyword), 77% (hybrid) and 91% (hybrid + reranker) of the top 20 search results come from the year asked.

## Repository contents

```
├── README.md
├── notebooks/      the nine Kaggle notebooks, with their saved outputs
├── Docs/           report and presentation slides
```

## Notebooks

Run them on Kaggle in this order. Each notebook's first cell explains what it does, what it needs and what it produces.

| # | Notebook | What it does | Inputs to attach | Time |
|---|---|---|---|---|
| 1 | `TRAG_01_data_prices_and_events` | Prices, abnormal returns, SEC events, Fed dates; downloads and cleans FNSPID news | none (Internet on) | 12 min |
| 2 | `TRAG_02_sentiment_embeddings_and_price_tests` | FinBERT sentiment, timelines, bge embeddings, time-blindness test, sentiment–price tests | Notebook 1 output | 8 min |
| 3 | `TRAG_03_first_trag_version_news_only` | First version of the system and its evaluation (142 questions, news only) | Notebook 2 output | 41 min |
| 4 | `TRAG_04_tweets_filings_and_knowledge_graph` | Tweets, press releases, XBRL revenue, Fed statements, policy actions, LLM event summaries, knowledge graph | Notebook 2 output; Kaggle dataset "Stock Tweets for Sentiment Analysis and Prediction" (by equinxx) | 22 min |
| 4J | `TRAG_04j_independent_llm_sentiment_judge` | Second, independent sentiment label for all 107,509 documents | Notebook 2 output; tweets dataset | 4.7 h |
| 5 | `TRAG_05_full_system_and_final_evaluation` | TempFinQA-v2, router, eight versions (B0–B6), scoring, sensitivity, structured reports | Notebook 4 output (+ Notebook 4J output) | 4.3 h |
| 6 | `TRAG_06_final_tables_and_figures` | Dual answer key, final tables and figures | Notebooks 4, 4J and 5 outputs | 5 min |
| 7 | `TRAG_07_search_methods_time_blindness_test` | Time-blindness test for every search method | Notebook 4 output | 1 min |
| 8 | `TRAG_08_live_demo_standard_rag_vs_trag` | Live demo: any question, as of any date, standard RAG vs T-RAG step by step | Notebook 4 output | 10 min |

Notebook 3 is the earlier version of the system; the final results come from Notebooks 5 and 6.

## How to reproduce

1. On Kaggle, create a new notebook and use **File → Import Notebook** to load a notebook from `notebooks/`.
2. **Settings :** turn **Internet on**. Notebooks 2, 3, 4, 4J, 5 and 8 need **GPU T4 ×2**; Notebooks 1 and 7 run on CPU; Notebook 6 runs on either.
3. **Inputs :** use **Add Input → Notebook Output Files** to attach the earlier notebooks listed in the table. The notebooks find their inputs by folder name (`trag_data`, `trag_index`, `trag_index_v2`, `trag_judge`, `trag_eval_v2_*`), so notebook names do not matter.
4. In Notebook 1, set `SEC_USER_AGENT` in the config cell to your own name and e-mail (the SEC requires a contact for automated requests).
5. Click **Save Version → Save & Run All**.

A fresh run downloads prices, filings and models again. Yahoo re-adjusts past prices for later dividends, and GPU arithmetic can occasionally change a generated answer, so a few numbers may differ slightly from ours. Our exact outputs are saved inside each notebook.

## Data sources

* **News :** FNSPID dataset (Hugging Face, `Zihan1004/FNSPID`)
* **Prices :** Yahoo Finance, via `yfinance` (split-adjusted; QQQ as the market)
* **Earnings :** SEC EDGAR 8-K filings (press releases) and XBRL company facts (reported revenue)
* **Policy :** Federal Reserve FOMC statements and 11 dated government policy actions
* **Social media :** Kaggle dataset "Stock Tweets for Sentiment Analysis and Prediction"

## Models

* **Router, answers, event summaries and sentiment judge :** Qwen2.5-7B-Instruct
* **Sentiment :** FinBERT (`ProsusAI/finbert`)
* **Search :** BM25 + `BAAI/bge-base-en-v1.5`, fused with Reciprocal Rank Fusion, re-ranked by `BAAI/bge-reranker-base`
* **Explanation check :** `cross-encoder/nli-deberta-v3-base`

All of them run on free Kaggle GPUs.
