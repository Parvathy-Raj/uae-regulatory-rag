"""Streamlit chat UI. Run: `make ui` (streamlit run src/uae_rag/ui/app.py).

Visual theme follows the Prompt Heist arena: warm brown/red/orange palette, dotted
background, gradient hero title and a chat "card" with a header bar.
"""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from uae_rag.config import RETRIEVAL_MODES, load_settings
from uae_rag.generate.answer import Answer
from uae_rag.llm.client import LLMError
from uae_rag.pipeline import RAGPipeline
from uae_rag.ui.render import (
    MODE_LABELS,
    answer_html,
    bubble,
    meta_html,
    page_label,
    sources_html,
)

LOGO = (Path(__file__).parent / "assets" / "logo.svg").read_text(encoding="utf-8")
EXAMPLES = [
    "When must a business register for VAT?",
    "What is a Qualifying Free Zone Person?",
    "Are natural persons subject to Corporate Tax?",
    "When is a Corporate Tax return due?",
]

CSS = """
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; }
.stApp { background:
  radial-gradient(900px 520px at 0% -5%, #F9DCCD 0%, rgba(249,220,205,0) 60%),
  radial-gradient(800px 480px at 100% 0%, #FDEBD9 0%, rgba(253,235,217,0) 55%),
  radial-gradient(#ECE3DD 1px, transparent 1px) 0 0 / 22px 22px,
  #FCF8F5 !important; font-family: 'Inter', system-ui, sans-serif; }
.block-container { max-width: 820px !important; padding: 28px 16px 40px !important; }

#hero { text-align: center; margin: 4px 0 18px; }
#hero .logo { width: 84px; height: 84px; margin: 0 auto; filter: drop-shadow(0 14px 26px rgba(179,74,68,.35));
  animation: bob 4s ease-in-out infinite; }
#hero .logo svg { width: 84px; height: 84px; }
@keyframes bob { 0%,100% { transform: translateY(0) } 50% { transform: translateY(-5px) } }
#hero h1 { font-size: 36px; font-weight: 800; letter-spacing: -.03em; margin: 14px 0 6px; padding: 0;
  background: linear-gradient(90deg, #60241E, #B34A44); -webkit-background-clip: text; background-clip: text; color: transparent; }
#hero p { margin: 0; color: #78716C; font-size: 15px; }
#hero .steps { display: flex; justify-content: center; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
#hero .steps span { background: rgba(255,255,255,.8); border: 1px solid #ECE3DD; border-radius: 999px;
  padding: 5px 12px 5px 6px; font-size: 13px; color: #44403C; }
#hero .steps b { display: inline-grid; place-items: center; width: 20px; height: 20px; margin-right: 6px;
  border-radius: 999px; background: #95271D; color: #fff; font-size: 11px; }

/* retriever picker = Prompt Heist level pills */
[data-testid="stButtonGroup"] { justify-content: center; }
[data-testid="stButtonGroup"] button { border-radius: 999px !important; border: 1px solid #ECE3DD !important;
  background: #fff !important; color: #44403C !important; font-weight: 500; box-shadow: none !important; margin: 0 4px; }
[data-testid="stButtonGroup"] button:hover { border-color: #EE9F80 !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] { background: #95271D !important; border-color: #95271D !important; color: #fff !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"] * { color: #fff !important; }

/* chat card */
.st-key-playcard { background: #fff; border-radius: 24px; overflow: hidden; gap: 0 !important;
  box-shadow: 0 1px 2px rgba(60,30,25,.04), 0 12px 40px rgba(96,36,30,.10); padding: 0 0 6px; margin-top: 10px; }
  background: linear-gradient(90deg, #FBE9E0, #FDF3EC); border-bottom: 1px solid #F1E7E1; }
  font-weight: 800; font-size: 16px; color: #fff; background: linear-gradient(135deg, #B34A44, #E77B49);
  box-shadow: 0 6px 16px rgba(179,74,68,.30); }

.st-key-thread { padding: 14px 16px 4px; gap: 10px !important; }
.msg { display: flex; }
.msg.user { justify-content: flex-end; }
.msg .bubble { max-width: 86%; padding: 10px 14px; border-radius: 18px; font-size: 15px; line-height: 1.55; }
.msg.bot .bubble { background: #F6EFEA; color: #2A1714; border-bottom-left-radius: 6px; }
.msg.user .bubble { background: #95271D; color: #fff; border-bottom-right-radius: 6px; }
.msg.refused .bubble { background: #FDF0E8; border: 1px dashed #EE9F80; }
.msg.error .bubble { background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; }
sup.cite { display: inline-grid; place-items: center; min-width: 17px; height: 17px; padding: 0 4px; margin-left: 2px;
  border-radius: 999px; background: #95271D; color: #fff; font-size: 10.5px; font-weight: 600; vertical-align: super; }
.srcs { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0 0 2px; }
.src { text-decoration: none !important; font-size: 12.5px; color: #95271D !important; background: #FDF0E8;
  border: 1px solid #F8DCCD; border-radius: 999px; padding: 3px 10px 3px 4px; }
.src:hover { background: #F8DCCD; }
.src b { display: inline-grid; place-items: center; width: 17px; height: 17px; margin-right: 5px; border-radius: 999px;
  background: #95271D; color: #fff; font-size: 10.5px; }
.meta { font-size: 12px; color: #A8A29E; margin: 4px 0 0 4px; }
.st-key-playcard [data-testid="stExpander"] { margin: 0 0 0 2px; border: none !important; }
.st-key-playcard [data-testid="stExpander"] summary { font-size: 13px; color: #A8A29E; }
.passage { font-size: 13.5px; color: #44403C; border-left: 3px solid #F8DCCD; padding: 2px 0 2px 10px; margin: 6px 0 12px; }
.passage .ph { font-size: 12px; color: #95271D; font-weight: 600; margin-bottom: 2px; }

/* example chips = Prompt Heist "tricks" */
.st-key-examples { padding: 8px 16px 0; }
.st-key-examples [data-testid="stButtonGroup"] { justify-content: flex-start; }
.st-key-examples [data-testid="stButtonGroup"] button { background: #FDF0E8 !important; border: 1px solid #F8DCCD !important;
  color: #95271D !important; font-size: 13px !important; margin: 0 6px 6px 0; }
.st-key-examples [data-testid="stButtonGroup"] button * { color: #95271D !important; }
.st-key-examples [data-testid="stButtonGroup"] button:hover { background: #F8DCCD !important; }

[data-testid="stChatInput"] { margin: 8px 16px 10px; border-radius: 14px !important; border: 1px solid #ECE3DD !important;
  background: #FBF7F4 !important; }
[data-testid="stChatInput"] textarea { font-size: 15px !important; }
[data-testid="stChatInputSubmitButton"] { background: #95271D !important; border-radius: 10px !important; }
[data-testid="stChatInputSubmitButton"] svg { fill: #fff !important; color: #fff !important; }
.footnote { text-align: center; color: #A8A29E; font-size: 12px; margin-top: 14px; }

@media (max-width: 640px) {
  #hero .logo, #hero .logo svg { width: 60px; height: 60px; }
  #hero h1 { font-size: 26px; margin: 10px 0 4px; }
  #hero p { font-size: 13.5px; padding: 0 8px; }
  #hero .steps span { font-size: 12px; padding: 4px 10px 4px 4px; }
  .msg .bubble { font-size: 14.5px; max-width: 92%; }
}
"""

HERO = f"""
<div id="hero">
  <div class="logo">{LOGO}</div>
  <h1>Tax Guide Desk</h1>
  <p>Ask about UAE VAT and Corporate Tax. Every answer is grounded in public FTA guides, with page citations.</p>
  <div class="steps">
    <span><b>1</b>Pick a retriever</span><span><b>2</b>Ask a tax question</span><span><b>3</b>Check the cited pages</span>
  </div>
</div>
"""


@st.cache_resource(show_spinner="Loading models (first run downloads ~400 MB)…")
def get_pipeline() -> RAGPipeline:
    pipe = RAGPipeline(load_settings())
    _ = pipe.chunks  # fail fast if the index is missing
    return pipe


def render_message(msg: dict) -> None:
    if msg["role"] == "user":
        st.markdown(bubble("user", html.escape(msg["content"])), unsafe_allow_html=True)
        return
    if "error" in msg:
        st.markdown(bubble("bot", html.escape(msg["error"]), "error"), unsafe_allow_html=True)
        return
    ans: Answer = msg["answer"]
    body = answer_html(ans.answer) + sources_html(ans) + meta_html(ans)
    st.markdown(bubble("bot", body, "refused" if ans.refused else ""), unsafe_allow_html=True)
    shown = [
        (i, s) for i, s in enumerate(ans.retrieved, 1) if s.chunk.chunk_id in set(ans.citations)
    ] or (list(enumerate(ans.retrieved, 1)) if ans.refused else [])
    if shown:
        label = "Cited passages" if not ans.refused else "Closest passages found"
        with st.expander(f"{label} ({len(shown)})"):
            for i, s in shown:
                c = s.chunk
                st.markdown(
                    f'<div class="passage"><div class="ph">[{i}] {html.escape(c.source_title)} · '
                    f"{page_label(c.page_start, c.page_end)} · {html.escape(c.heading)}</div>"
                    f"{html.escape(c.text)}</div>",
                    unsafe_allow_html=True,
                )


def queue_example() -> None:
    choice = st.session_state.get("example_pick")
    if choice:
        st.session_state.pending = choice
        st.session_state.example_pick = None


def main() -> None:
    st.set_page_config(page_title="Tax Guide Desk", page_icon="📑", layout="centered")
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    st.markdown(HERO, unsafe_allow_html=True)
    st.session_state.setdefault("messages", [])

    settings = load_settings()
    mode = (
        st.segmented_control(
            "Retriever",
            options=list(RETRIEVAL_MODES),
            format_func=MODE_LABELS.get,
            default=settings.retrieval.mode,
            key="mode",
            label_visibility="collapsed",
        )
        or settings.retrieval.mode
    )

    try:
        pipe = get_pipeline()
    except FileNotFoundError:
        st.warning(
            "The document index isn't built yet. Run `make download` then `make ingest`, and reload."
        )
        st.stop()

    with st.container(key="playcard"):
        with st.container(key="thread"):
            st.markdown(
                bubble(
                    "bot",
                    "Hi! Ask me anything about UAE VAT or Corporate Tax. I'll answer only from the "
                    "public FTA guides and show you the pages I used.",
                ),
                unsafe_allow_html=True,
            )
            for msg in st.session_state.messages:
                render_message(msg)
        with st.container(key="examples"):
            st.pills(
                "Try",
                EXAMPLES,
                key="example_pick",
                on_change=queue_example,
                label_visibility="collapsed",
            )
        typed = st.chat_input("Ask about VAT or Corporate Tax…", max_chars=800)

    question = typed or st.session_state.pop("pending", None)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            with st.spinner("Searching the guides…"):
                ans = pipe.ask(question, mode)
            st.session_state.messages.append({"role": "assistant", "answer": ans})
        except LLMError as exc:
            st.session_state.messages.append({"role": "assistant", "error": f"LLM error: {exc}"})
        st.rerun()

    st.markdown(
        '<div class="footnote">Unofficial demo built on public FTA guides. Not tax advice. '
        "Always check the source document.</div>",
        unsafe_allow_html=True,
    )


main()
