"""
app.py
------
Streamlit web interface for the Digital Financial Literacy AI Agent.
Uses local multilingual embeddings and a persistent ChromaDB knowledge base.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from rag_engine import (
    LANGUAGE_HINTS,
    FinancialLiteracyRAG,
)

# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Financial Literacy Agent",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Main background */
        .main { background-color: #f0f4f8; }

        /* Chat bubbles */
        .chat-user {
            background: #1a73e8;
            color: #ffffff;
            padding: 10px 16px;
            border-radius: 18px 18px 4px 18px;
            margin: 6px 0 6px auto;
            max-width: 78%;
            width: fit-content;
            font-size: 15px;
            line-height: 1.5;
        }
        .chat-assistant {
            background: #ffffff;
            color: #1f2328;
            padding: 14px 18px;
            border-radius: 18px 18px 18px 4px;
            margin: 6px auto 6px 0;
            max-width: 82%;
            width: fit-content;
            font-size: 15px;
            line-height: 1.7;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.07);
            white-space: pre-wrap;
        }
        .chat-assistant ul {
            margin: 6px 0 6px 18px;
            padding: 0;
        }
        .chat-assistant li { margin-bottom: 4px; }
        .chat-container { padding: 8px 0; }

        /* Sidebar — white background, black text */
        section[data-testid="stSidebar"] { background-color: #ffffff; }

        /* All sidebar text — black */
        section[data-testid="stSidebar"] *,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] li,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span { color: #1a1a1a !important; }

        /* Selectbox — white background, black text */
        section[data-testid="stSidebar"] [data-baseweb="select"] > div { background: #ffffff !important; }
        section[data-testid="stSidebar"] [data-baseweb="select"] * { color: #1a1a1a !important; }

        /* Divider */
        section[data-testid="stSidebar"] hr { border-color: #d1d5db; }

        /* Suggestion pills */
        div[data-testid="stButton"] > button {
            background: #f1f5f9;
            border: 1px solid #cbd5e1;
            border-radius: 20px;
            font-size: 13px;
            padding: 4px 14px;
            color: #334155;
        }
        div[data-testid="stButton"] > button:hover {
            background: #e2e8f0;
        }

        /* Header banner */
        .app-header {
            background: linear-gradient(135deg, #1e3a5f 0%, #1a73e8 100%);
            padding: 20px 28px;
            border-radius: 12px;
            margin-bottom: 18px;
            color: white;
        }
        .app-header h1 { margin: 0; font-size: 26px; color: white; }
        .app-header p  { margin: 4px 0 0; font-size: 14px; color: #c7d9f5; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    language = st.selectbox(
        "🌐 Language / भाषा / மொழி",
        options=list(LANGUAGE_HINTS.keys()),
        index=0,
        format_func=lambda name: f"{name} · {LANGUAGE_HINTS[name]}",
        help="Suggested questions and answers use this language.",
    )

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    with st.expander("📚 Resources & documents"):
        st.link_button("RBI financial education", "https://www.rbi.org.in/FinancialEducation/")
        st.link_button("SEBI investor education", "https://investor.sebi.gov.in/")
        st.link_button("Report cyber fraud · 1930", "https://cybercrime.gov.in/")
        uploaded_files = st.file_uploader(
            "Add a PDF or TXT guide",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            docs_path = Path("./financial_docs")
            docs_path.mkdir(exist_ok=True)
            for uploaded_file in uploaded_files:
                dest = docs_path / Path(uploaded_file.name).name
                dest.write_bytes(uploaded_file.getvalue())
            st.caption(f"{len(uploaded_files)} file(s) ready to add.")
            if st.button("Add to knowledge base", use_container_width=True):
                with st.spinner("Processing documents…"):
                    try:
                        if "rag" not in st.session_state:
                            st.session_state["rag"] = FinancialLiteracyRAG()
                        st.session_state["rag"].reingest()
                    except Exception as exc:
                        st.error(f"Could not add documents: {exc}")
                    else:
                        st.success("Documents added.")

# ── Session state initialisation ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "rag" not in st.session_state:
    with st.spinner("🚀 Loading AI engine (first load may take ~30 s)…"):
        try:
            st.session_state["rag"] = FinancialLiteracyRAG()
        except Exception as exc:
            st.error(
                f"**Failed to initialise the RAG engine.**\n\n{exc}\n\n"
                "Check your installed dependencies and network access for the first model download."
            )
            st.stop()

rag: FinancialLiteracyRAG = st.session_state["rag"]

# ── Header banner ──────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-header">
        <h1>💰 Digital Financial Literacy Agent</h1>
        <p>Learn about UPI, online scams, loans, budgeting, and government schemes — in your language.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Localized prompts keep the example questions aligned with answer language ─
SUGGESTIONS_BY_LANGUAGE = {
    "English": [
        "How do I send money using UPI?",
        "How can I avoid online scams?",
        "Explain the 50-30-20 budget rule.",
        "How do I compare loan interest rates?",
    ],
    "Hindi": [
        "UPI से पैसे कैसे भेजें?",
        "ऑनलाइन धोखाधड़ी से कैसे बचें?",
        "50-30-20 बजट नियम समझाएँ।",
        "ऋण की ब्याज दरों की तुलना कैसे करें?",
    ],
    "Tamil": [
        "UPI மூலம் பணம் அனுப்புவது எப்படி?",
        "ஆன்லைன் மோசடிகளைத் தவிர்ப்பது எப்படி?",
        "50-30-20 பட்ஜெட் விதியை விளக்குங்கள்.",
        "கடன் வட்டி விகிதங்களை ஒப்பிடுவது எப்படி?",
    ],
    "Telugu": [
        "UPI ద్వారా డబ్బు ఎలా పంపాలి?",
        "ఆన్‌లైన్ మోసాలను ఎలా నివారించాలి?",
        "50-30-20 బడ్జెట్ నియమాన్ని వివరించండి.",
        "రుణ వడ్డీ రేట్లను ఎలా పోల్చాలి?",
    ],
    "Bengali": [
        "UPI দিয়ে টাকা পাঠাব কীভাবে?",
        "অনলাইন জালিয়াতি এড়াব কীভাবে?",
        "৫০-৩০-২০ বাজেটের নিয়মটি বুঝিয়ে বলুন।",
        "ঋণের সুদের হার তুলনা করব কীভাবে?",
    ],
    "Marathi": [
        "UPI द्वारे पैसे कसे पाठवायचे?",
        "ऑनलाइन फसवणूक कशी टाळायची?",
        "५०-३०-२० बजेटचा नियम समजावून सांगा.",
        "कर्जाच्या व्याजदरांची तुलना कशी करायची?",
    ],
}
LOCALIZED_UI = {
    "English": ("Try asking", "Ask a question about UPI, scams, budgeting, or loans…"),
    "Hindi": ("पूछकर देखें", "UPI, धोखाधड़ी, बजट या ऋण के बारे में पूछें…"),
    "Tamil": ("கேட்டுப் பாருங்கள்", "UPI, மோசடி, பட்ஜெட் அல்லது கடன் பற்றி கேளுங்கள்…"),
    "Telugu": ("ఇలా అడగండి", "UPI, మోసాలు, బడ్జెట్ లేదా రుణాల గురించి అడగండి…"),
    "Bengali": ("জিজ্ঞেস করে দেখুন", "UPI, জালিয়াতি, বাজেট বা ঋণ সম্পর্কে জিজ্ঞেস করুন…"),
    "Marathi": ("विचारून पाहा", "UPI, फसवणूक, बजेट किंवा कर्जाबद्दल विचारा…"),
}

suggestion_heading, input_placeholder = LOCALIZED_UI[language]
st.markdown(f"**💡 {suggestion_heading}**")
cols = st.columns(2)
for idx, suggestion in enumerate(SUGGESTIONS_BY_LANGUAGE[language]):
    if cols[idx % len(cols)].button(suggestion, key=f"sug_{idx}"):
        st.session_state["pending_question"] = suggestion

# ── Chat history ───────────────────────────────────────────────────────────
chat_container = st.container()

def _render_answer(text: str) -> str:
    """Convert plain-text answer (bullets, numbered lines, headers) to HTML."""
    import re as _re
    lines = text.splitlines()
    html_lines = []
    in_list = False
    numbered = _re.compile(r"^\d+\.")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            continue
        if stripped.startswith("• "):
            if not in_list:
                html_lines.append("<ul style='margin:6px 0 6px 18px;padding:0'>")
                in_list = True
            html_lines.append(
                f"<li style='margin-bottom:4px'>{html.escape(stripped[2:])}</li>"
            )
        elif numbered.match(stripped):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<p style='margin:3px 0'>{html.escape(stripped)}</p>")
        elif stripped.endswith(":") and len(stripped) < 80:
            # Section header line (e.g. "यहाँ आपके प्रश्न का उत्तर है:")
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(
                f"<p style='margin:6px 0 2px;font-weight:600'>{html.escape(stripped)}</p>"
            )
        elif stripped.startswith("📝"):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(
                "<p style='margin:10px 0 0;font-size:13px;color:#64748b'>"
                f"{html.escape(stripped)}</p>"
            )
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<p style='margin:4px 0'>{html.escape(stripped)}</p>")
    if in_list:
        html_lines.append("</ul>")
    return "".join(html_lines)


with chat_container:
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-container">'
                f'<div class="chat-user">🧑 {html.escape(msg["content"])}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-container">'
                f'<div class="chat-assistant">🤖 {_render_answer(msg["content"])}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

# ── Chat input ─────────────────────────────────────────────────────────────
user_input: str = st.chat_input(
    input_placeholder
)

# Merge suggestion click into user input
if "pending_question" in st.session_state:
    user_input = st.session_state.pop("pending_question")

# ── Process user message ───────────────────────────────────────────────────
if user_input and user_input.strip():
    question = user_input.strip()

    # Store user message
    st.session_state["messages"].append({"role": "user", "content": question})

    # Show user bubble immediately
    st.markdown(
        f'<div class="chat-container">'
        f'<div class="chat-user">🧑 {html.escape(question)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Generate answer
    with st.spinner("🤔 Thinking…"):
        try:
            answer = rag.answer(question=question, language=language)
        except Exception as exc:
            st.error(f"Could not retrieve an answer: {exc}")
        else:
            st.session_state["messages"].append(
                {"role": "assistant", "content": answer}
            )
            st.markdown(
                f'<div class="chat-container">'
                f'<div class="chat-assistant">🤖 {_render_answer(answer)}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


# ── Empty state ────────────────────────────────────────────────────────────
if not st.session_state["messages"]:
    st.markdown(
        """
        <div style="text-align:center; padding: 40px 0; color: #94a3b8;">
            <div style="font-size:56px;">💬</div>
            <p style="font-size:18px; margin-top:12px;">
                Start a conversation by typing a question below<br>or clicking one of the suggestions above.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <hr style="margin-top:40px; border-color:#e2e8f0;" />
    <div style="text-align:center; font-size:12px; color:#94a3b8; padding-bottom:12px;">
        Digital Financial Literacy Agent · Local retrieval-augmented answers ·
        Educational information, not personalized financial advice
    </div>
    """,
    unsafe_allow_html=True,
)
