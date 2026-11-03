"""Streamlit front end.   Run: uv run streamlit run app.py

Needs data/secom.db (uv run python -m fabpqagent.data) and data/model.joblib
(uv run python -m fabpqagent.model).
"""
from __future__ import annotations

import streamlit as st

from fabpqagent.data import ROOT

MODEL_PATH = ROOT / "data" / "model.joblib"
EXAMPLES = [
    "Lot 2 failed. Which sensors look most unusual on it?",
    "Is sensor s059 in statistical control?",
    "The fail rate seems high recently. Is it, and what is the likely cause?",
]

st.set_page_config(page_title="Fab PQ Agent", page_icon="🔬", layout="wide")
st.title("🔬 Fab product-quality troubleshooting agent")
st.caption("SECOM fab data (1,567 lots, 590 sensors) · fail classifier · SPC · BM25 notes · tool-calling agent")


@st.cache_resource
def build_context():
    from fabpqagent import db, model
    from fabpqagent.retrieval import NoteIndex, load_notes
    from fabpqagent.tools import ToolContext
    return ToolContext(lots=db.load_lots(), model=model.load(MODEL_PATH), notes=NoteIndex(load_notes()))


@st.cache_resource
def build_llm():
    from fabpqagent.llm import OpenRouterLLM
    return OpenRouterLLM()


def render(inv) -> None:
    st.subheader("Report")
    st.write(inv.report or "_(no report)_")
    cols = st.columns(4)
    cols[0].metric("Tool calls", len(inv.steps))
    cols[1].metric("Tokens", inv.prompt_tokens + inv.completion_tokens)
    cols[2].metric("Latency", f"{inv.latency_s:.1f}s")
    cols[3].metric("Stopped", inv.stop_reason)
    with st.expander("Investigator evidence"):
        st.markdown(inv.evidence or "_(none)_")
    with st.expander(f"Tool trace ({len(inv.steps)} steps)"):
        for i, s in enumerate(inv.steps, 1):
            st.markdown(f"**{i}. `{s.tool}`** `{s.arguments}`")
            st.code(s.observation, language="text")
    if inv.models:
        st.caption("Models used: " + ", ".join(dict.fromkeys(inv.models)))


question = st.text_input("Ask about lots, sensors or fail rates", placeholder=EXAMPLES[0])
example_cols = st.columns(len(EXAMPLES))
for col, ex in zip(example_cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        question = ex

if question:
    try:
        from fabpqagent.agent import investigate
        ctx, llm = build_context(), build_llm()
        with st.spinner("Investigating..."):
            render(investigate(question, llm, ctx))
    except NotImplementedError as err:
        st.warning(f"Not built yet: {err or 'a stub still raises NotImplementedError'}.")
    except FileNotFoundError as err:
        st.error(str(err))
    except Exception as err:  # show the failure in the UI instead of a blank page
        st.exception(err)
