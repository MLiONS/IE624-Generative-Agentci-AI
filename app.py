"""
app.py — Gradio chat interface for the Intelligent Academic Assistant (§5.6).

Run:   python app.py --share            (in Colab, or call launch_app(assistant) from the notebook)

• Single-document mode: upload ONE PDF → the index is built automatically → suggested
  questions appear as clickable buttons (NotebookLM-style). Uploading another PDF
  replaces the previous one and starts a new chat. "↻ More suggestions" = fresh set.
• Source panel: document, page, similarity, supporting excerpt.
• Settings: top-k, abstention threshold, compare with no-retrieval baseline.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rag_core import (AcademicAssistant, RAGConfig, build_or_load_index, list_pdfs,
                      format_answer_with_citations, format_sources_markdown, set_single_pdf)

MAX_SUGGESTIONS = 8                       # number of button slots in the UI
EMPTY_SOURCES = "_Sources for the latest answer appear here._"


def launch_app(assistant: AcademicAssistant, pdf_dir: str = "documents",
               index_dir: str = "index", share: bool = False):
    import gradio as gr

    state = {"seed": 0}

    # ------------------------------------------------------------------ helpers
    def doc_summary():
        if assistant.index is None:
            return "_No document yet. Upload a PDF above._"
        d = assistant.index.documents[0]
        cfg = assistant.config
        text = (f"📄 **{d['source']}** · {d['pages']} pages · {d['chunks']} chunks "
                f"(chunk {cfg.chunk_size_words} words, overlap {cfg.chunk_overlap_words})")
        if d["pages"] and len(d.get("skipped_pages", [])) > 0.3 * d["pages"]:
            text += ("\n\n⚠️ Most pages have no readable text (scanned or Turnitin copy?). "
                     "Upload the original PDF.")
        return text

    def suggestion_updates(questions):
        ups = [gr.update(value=q, visible=True) for q in questions[:MAX_SUGGESTIONS]]
        ups += [gr.update(value="", visible=False)] * (MAX_SUGGESTIONS - len(ups))
        return ups

    def load_suggestions(refresh=False):
        if assistant.index is None:
            return ["_Upload a PDF to see questions you can ask about it._"] + \
                suggestion_updates([])
        if refresh:
            state["seed"] += 1
        try:
            qs = assistant.suggested_questions(refresh=refresh, seed=state["seed"])
        except Exception as e:
            return [f"_Could not generate suggestions: {e}_"] + suggestion_updates([])
        return (["**💡 Questions you can ask about this PDF** — click one to ask it"]
                + suggestion_updates(qs))

    # ------------------------------------------------------------------ chat
    def respond(message, history, k, threshold, compare):
        history = list(history or [])
        if not message or not str(message).strip():
            return history, gr.update(), ""
        history.append({"role": "user", "content": message})
        if assistant.index is None:
            history.append({"role": "assistant",
                            "content": "Please upload a PDF first."})
            return history, EMPTY_SOURCES, ""
        ans = assistant.ask(message, k=int(k), min_similarity=float(threshold))
        reply = format_answer_with_citations(ans)
        if compare:
            base = assistant.ask_baseline(message)
            reply += f"\n\n---\n**Baseline (no retrieval):** {base.answer}"
        history.append({"role": "assistant", "content": reply})
        return history, format_sources_markdown(ans), ""

    # ------------------------------------------------------------------ documents
    def index_document(file, rebuild=False):
        """Upload → replace the current PDF → rebuild index → new suggestions, new chat."""
        try:
            if file is not None:
                path = file if isinstance(file, str) else file.name
                set_single_pdf(path, pdf_dir, index_dir)
            if not list_pdfs(pdf_dir):
                assistant.index = None
                return [doc_summary()] + load_suggestions() + [gr.update(), gr.update()]
            assistant.index = build_or_load_index(pdf_dir, index_dir, assistant.config,
                                                  rebuild=rebuild or file is not None)
        except Exception as e:
            return [f"❌ {e}"] + load_suggestions() + [gr.update(), gr.update()]
        state["seed"] = 0
        return [doc_summary()] + load_suggestions() + [[], EMPTY_SOURCES]

    def remove_document():
        shutil.rmtree(pdf_dir, ignore_errors=True)
        shutil.rmtree(index_dir, ignore_errors=True)
        assistant.index = None
        return [doc_summary()] + load_suggestions() + [[], EMPTY_SOURCES]

    # ------------------------------------------------------------------ layout
    try:
        chatbot = gr.Chatbot(type="messages", height=500, label="Academic Assistant")
    except TypeError:               # Gradio ≥6 uses the messages format by default
        chatbot = gr.Chatbot(height=500, label="Academic Assistant")

    with gr.Blocks(title="Academic Assistant (RAG)") as demo:
        gr.Markdown("## 📚 Intelligent Academic Assistant\n"
                    "Upload one PDF, then click a suggested question or type your own. "
                    "Answers are generated only from passages of that PDF and cite the page. "
                    "**Always check the cited passage.**")
        with gr.Row():
            with gr.Column(scale=3):
                upload = gr.File(file_count="single", file_types=[".pdf"],
                                 label="Upload ONE PDF (text-based) — indexing starts "
                                       "automatically; a new upload replaces the old one")
                docs_md = gr.Markdown(doc_summary())
                sugg_title = gr.Markdown()
                sugg_btns = []
                with gr.Row():
                    for i in range(MAX_SUGGESTIONS):
                        sugg_btns.append(gr.Button("", visible=False, size="sm",
                                                   variant="secondary", min_width=200))
                more_btn = gr.Button("↻ More suggestions", size="sm")
                chatbot.render()
                with gr.Row():
                    msg = gr.Textbox(placeholder="Ask a question about this PDF…",
                                     show_label=False, scale=5)
                    send = gr.Button("Ask", variant="primary", scale=1)
                clear = gr.Button("Clear chat")
            with gr.Column(scale=2):
                sources = gr.Markdown(EMPTY_SOURCES, label="Sources")

        with gr.Accordion("Settings", open=False):
            with gr.Row():
                k = gr.Slider(1, 8, value=assistant.config.top_k, step=1,
                              label="Top-k passages retrieved")
                thr = gr.Slider(0.0, 0.8, value=assistant.config.min_similarity, step=0.05,
                                label="Min. similarity to answer (abstention gate)")
                compare = gr.Checkbox(False, label="Also show no-retrieval baseline answer")
            clear_docs = gr.Button("🗑 Remove document", variant="stop")

        # ------------------------------------------------------------------ events
        chat_inputs = [chatbot, k, thr, compare]
        send.click(respond, [msg] + chat_inputs, [chatbot, sources, msg])
        msg.submit(respond, [msg] + chat_inputs, [chatbot, sources, msg])
        for b in sugg_btns:          # clicking a suggestion asks it (button value = question)
            b.click(respond, [b] + chat_inputs, [chatbot, sources, msg])
        more_btn.click(lambda: load_suggestions(refresh=True), None, [sugg_title] + sugg_btns)
        clear.click(lambda: ([], EMPTY_SOURCES), None, [chatbot, sources])

        doc_outputs = [docs_md, sugg_title] + sugg_btns + [chatbot, sources]
        upload.upload(index_document, [upload], doc_outputs)
        clear_docs.click(remove_document, None, doc_outputs)
        demo.load(lambda: load_suggestions(), None, [sugg_title] + sugg_btns)

    demo.queue().launch(share=share, debug=False)
    return demo


def console_loop(assistant: AcademicAssistant):
    """Notebook/terminal fallback if Gradio is unavailable (§5.6)."""
    print("Academic Assistant — type a question, a suggestion number, or 'quit'.")
    sugg = assistant.suggested_questions()
    for i, q in enumerate(sugg, 1):
        print(f"  {i}. {q}")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in {"quit", "exit", "q", ""}:
            break
        if q.isdigit() and 1 <= int(q) <= len(sugg):
            q = sugg[int(q) - 1]
            print("→", q)
        ans = assistant.ask(q)
        print("\nAnswer:", ans.answer)
        for s in ans.sources:
            print(f"  [{s['rank']}] {s['source']} {s['pages']}  (sim {s['score']:.2f})")
        print(f"  ({ans.seconds:.2f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", help="the one PDF to ask questions about")
    ap.add_argument("--pdf_dir", default="documents")
    ap.add_argument("--index_dir", default="index")
    ap.add_argument("--generator", default=RAGConfig.generator_model)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--share", action="store_true", help="public Gradio link (Colab)")
    ap.add_argument("--console", action="store_true", help="text loop instead of Gradio")
    args = ap.parse_args()

    cfg = RAGConfig(generator_model=args.generator)
    if args.pdf:
        set_single_pdf(args.pdf, args.pdf_dir, args.index_dir)
    idx = None
    if Path(args.pdf_dir).exists() and list_pdfs(args.pdf_dir):
        idx = build_or_load_index(args.pdf_dir, args.index_dir, cfg, rebuild=args.rebuild)
    assistant = AcademicAssistant(idx, cfg)
    if args.console:
        if idx is None:
            raise SystemExit(f"Put PDFs in {args.pdf_dir}/ first.")
        console_loop(assistant)
    else:
        launch_app(assistant, args.pdf_dir, args.index_dir, share=args.share)
