"""
rag_engine.py
-------------
Zero-LangChain RAG pipeline for the Digital Financial Literacy Agent.

Stack
-----
  • Document loading  – pure Python (pypdf for PDF, built-in open() for TXT)
  • Chunking          – custom RecursiveCharacterSplitter (no external deps)
  • Embeddings        – sentence-transformers (multilingual MiniLM)
  • Vector store      – chromadb (persistent, local)
  • Responses         – grounded retrieval, with optional Watsonx generation
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ── Embeddings ─────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer

# ── Vector store ───────────────────────────────────────────────────────────
import chromadb
from chromadb.config import Settings

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

CHROMA_PERSIST_DIR = "./chroma_db"
DOCS_DIR           = "./financial_docs"
COLLECTION_NAME    = "financial_literacy"
NON_CONSUMER_FINANCE_DOCS = {"UpdatedGFR31July2025_0.pdf"}

# Multilingual MiniLM — handles EN, HI, TA, TE, BN, MR and 50+ more languages
EMBEDDING_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE    = 600   # characters
CHUNK_OVERLAP = 80
SEED_VERSION  = 8
MAX_RETRIEVAL_DISTANCE = 0.60
RETRIEVAL_DISTANCE_MARGIN = 0.12
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct")
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY") or os.getenv("IBM_CLOUD_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
_generation_model = None

PAGE_LABEL_LINE = re.compile(
    r"(?im)^\s*(?:page|pg\.?|p\.)\s*\d+(?:\s*(?:of|/)\s*\d+)?\s*$"
)
PAGE_REFERENCE = re.compile(
    r"(?i)(?:\(\s*(?:page|pg\.?|p\.)\s*\d+(?:\s*(?:of|/)\s*\d+)?\s*\)"
    r"|\[\s*(?:page|pg\.?|p\.)\s*\d+(?:\s*(?:of|/)\s*\d+)?\s*\]"
    r"|\b(?:page|pg\.?|p\.)\s*\d+(?:\s*(?:of|/)\s*\d+)?\b"
    r"|\b\d{1,4}\s*\|\s*p\s*a\s*g\s*e\b)"
)
INLINE_TOC_CHAPTER = re.compile(
    r"(?i)\s+\d{1,3}\s+chapter\s+\d+\b[^\n]*"
)
TOC_CHAPTER_LINE = re.compile(
    r"(?i)^\s*(?:\d{1,3}\s+)?chapter\s+\d+\b.*"
    r"(?:\.{2,}|…|\s+\d{1,3}\s*$).*$"
)


def _remove_page_references(text: str) -> str:
    """Remove page labels and inline page references from extracted text."""
    text = PAGE_LABEL_LINE.sub("", text)
    text = PAGE_REFERENCE.sub("", text)
    text = INLINE_TOC_CHAPTER.sub("", text)
    text = "\n".join(
        line for line in text.splitlines() if not TOC_CHAPTER_LINE.match(line)
    )
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"[ \t]+\n", "\n", text).strip()


def _get_generation_model():
    """Return a lazily initialized Watsonx model, or None when not configured."""
    global _generation_model
    if not WATSONX_APIKEY or not WATSONX_PROJECT_ID:
        return None
    if _generation_model is None:
        from ibm_watsonx_ai.foundation_models import Model

        _generation_model = Model(
            model_id=WATSONX_MODEL_ID,
            credentials={"apikey": WATSONX_APIKEY, "url": WATSONX_URL},
            project_id=WATSONX_PROJECT_ID,
            params={
                "decoding_method": "greedy",
                "max_new_tokens": 300,
                "min_new_tokens": 1,
                "repetition_penalty": 1.05,
            },
        )
    return _generation_model


def generation_is_configured() -> bool:
    """Whether credentials are present for conversational Watsonx answers."""
    return bool(WATSONX_APIKEY and WATSONX_PROJECT_ID)


def _generate_answer(question: str, context: str, language: str) -> str:
    """Generate a concise answer grounded only in retrieved source material."""
    model = _get_generation_model()
    if model is None:
        return _format_chunks([context])

    prompt = (
        "You are a friendly digital financial literacy assistant. Answer the user's "
        f"question in {language}. Use only the reference material below; do not make "
        "up facts, rates, guarantees, or instructions. If the references do not answer "
        "the question, say so briefly. Explain clearly in a natural conversational "
        "style, using short paragraphs or a few bullets where helpful. Do not mention "
        "page numbers, page references, or claim to be a financial adviser. "
        "This is general educational information, not personalized financial advice.\n\n"
        f"Reference material:\n{context}\n\n"
        f"User question: {question}\n"
        "Answer:"
    )
    response = model.generate_text(prompt=prompt)
    if not isinstance(response, str) or not response.strip():
        raise RuntimeError("Watsonx returned an empty or invalid answer.")
    answer = _remove_page_references(response)
    if not answer:
        raise RuntimeError("Watsonx returned an empty answer after cleanup.")
    return answer

# Language display names (used by app.py selectbox)
LANGUAGE_HINTS: dict[str, str] = {
    "English": "English",
    "Hindi":   "हिन्दी",
    "Tamil":   "தமிழ்",
    "Telugu":  "తెలుగు",
    "Bengali": "বাংলা",
    "Marathi": "मराठी",
}

# "No result" messages per language
NO_RESULT_MSG: dict[str, str] = {
    "English": "I could not find relevant information for your question.",
    "Hindi":   "आपके प्रश्न के लिए प्रासंगिक जानकारी नहीं मिली।",
    "Tamil":   "உங்கள் கேள்விக்கு தொடர்புடைய தகவல் கிடைக்கவில்லை.",
    "Telugu":  "మీ ప్రశ్నకు సంబంధిత సమాచారం దొరకలేదు.",
    "Bengali": "আপনার প্রশ্নের জন্য প্রাসঙ্গিক তথ্য পাওয়া যায়নি।",
    "Marathi": "आपल्या प्रश्नासाठी संबंधित माहिती सापडली नाही.",
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. Document loading (no LangChain)
# ══════════════════════════════════════════════════════════════════════════════

def _load_txt(path: Path) -> str:
    """Read a plain-text file."""
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: Path) -> str:
    """Extract text from a PDF using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        return "\n".join(
            _remove_page_references(page.extract_text() or "")
            for page in reader.pages
        )
    except Exception as exc:
        print(f"[rag_engine] Could not read PDF {path.name}: {exc}")
        return ""


def load_documents_from_dir(docs_dir: str = DOCS_DIR) -> list[dict]:
    """
    Walk *docs_dir* and return a list of dicts:
        {"text": <str>, "source": <filename>, "language": <lang>}
    Supports .txt and .pdf files.
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        docs_path.mkdir(parents=True)

    records: list[dict] = []
    for file in sorted(docs_path.rglob("*")):
        # Government financial rules concern public-sector expenditure, not consumer finance.
        if file.name in NON_CONSUMER_FINANCE_DOCS:
            continue
        if file.suffix.lower() == ".txt" and file.name != "README.txt":
            text = _load_txt(file)
        elif file.suffix.lower() == ".pdf":
            text = _load_pdf(file)
        else:
            continue
        if text.strip():
            records.append({"text": text, "source": file.name, "language": "English"})

    return records


# ══════════════════════════════════════════════════════════════════════════════
# 2. Text chunker (no LangChain)
# ══════════════════════════════════════════════════════════════════════════════

def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Recursively split *text* on paragraph → sentence → word boundaries,
    targeting chunks of ~*chunk_size* characters with *overlap* carry-over.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks: list[str] = []

    def _split(fragment: str, seps: list[str]) -> list[str]:
        if not seps or len(fragment) <= chunk_size:
            return [fragment] if fragment.strip() else []
        sep = seps[0]
        parts = re.split(re.escape(sep), fragment) if sep else list(fragment)
        result: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).lstrip(sep) if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    result.append(current.strip())
                if len(part) > chunk_size:
                    result.extend(_split(part, seps[1:]))
                    current = ""
                else:
                    current = part
        if current.strip():
            result.append(current.strip())
        return result

    raw_chunks = _split(text, separators)

    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            chunks.append(chunk)
        else:
            available_overlap = max(0, chunk_size - len(chunk) - 1)
            overlap_size = min(overlap, available_overlap)
            tail = chunks[-1][-overlap_size:] if overlap_size else ""
            chunks.append(f"{tail} {chunk}".strip() if tail else chunk)

    return chunks


def chunk_documents(records: list[dict]) -> list[dict]:
    """
    Take raw document records and return a flat list of chunk dicts:
        {"text": <str>, "source": <filename>, "language": <lang>, "chunk_id": <uuid>}
    """
    chunks: list[dict] = []
    for rec in records:
        lang = rec.get("language", "English")
        for chunk_text in _split_text(rec["text"]):
            chunks.append({
                "text":     chunk_text,
                "source":   rec["source"],
                "language": lang,
                "chunk_id": str(uuid.uuid4()),
            })
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 3. Embeddings (sentence-transformers)
# ══════════════════════════════════════════════════════════════════════════════

_embedding_model: Optional[SentenceTransformer] = None

def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return model.encode(texts, normalize_embeddings=True).tolist()


# ══════════════════════════════════════════════════════════════════════════════
# 4. ChromaDB vector store
# ══════════════════════════════════════════════════════════════════════════════

def _get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "seed_version": SEED_VERSION},
    )


def ingest_documents(docs_dir: str = DOCS_DIR) -> chromadb.Collection:
    """
    Load documents from *docs_dir* AND always include the built-in multilingual
    seed knowledge. This ensures non-English language queries always work even
    when user uploads contain only English PDFs.
    Returns the ChromaDB collection.
    """
    user_records = load_documents_from_dir(docs_dir)
    seed_records = _get_seeded_records()

    # Always merge seed (multilingual) + user docs
    records = seed_records + user_records

    chunks = chunk_documents(records)
    collection = _get_chroma_collection()

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch      = chunks[i : i + batch_size]
        ids        = [c["chunk_id"] for c in batch]
        texts      = [c["text"]     for c in batch]
        metadatas  = [{"source": c["source"], "language": c.get("language", "English")} for c in batch]
        embeddings = embed_texts(texts)
        collection.upsert(ids=ids, documents=texts,
                          metadatas=metadatas, embeddings=embeddings)

    return collection


def get_or_create_collection(docs_dir: str = DOCS_DIR) -> chromadb.Collection:
    """
    Return the ChromaDB collection, creating/populating it if needed.
    Rebuild it when the built-in knowledge base changes.
    """
    collection = _get_chroma_collection()
    if collection.count() == 0:
        return ingest_documents(docs_dir)

    if collection.metadata.get("seed_version") != SEED_VERSION:
        client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        client.delete_collection(COLLECTION_NAME)
        return ingest_documents(docs_dir)

    return collection


# ══════════════════════════════════════════════════════════════════════════════
# 5. Answer synthesis (offline — formats retrieved chunks into a readable answer)
# ══════════════════════════════════════════════════════════════════════════════

_NUMBERED = re.compile(r"^\d+\.")


def _format_chunks(chunks: list[str]) -> str:
    """
    Format a list of retrieved text chunks into clean HTML-ready plain text.
    Numbered lines (1. 2. …) are kept as-is; all other non-empty lines get a bullet.
    Adjacent numbered lines are rendered as a flat numbered list.
    """
    lines: list[str] = []
    for chunk in chunks:
        chunk = _remove_page_references(chunk)
        for line in chunk.splitlines():
            line = line.strip()
            if line:
                lines.append(line)

    body: list[str] = []
    for line in lines:
        if _NUMBERED.match(line):
            body.append(line)
        else:
            body.append(f"• {line}")
    return "\n".join(body)


# ══════════════════════════════════════════════════════════════════════════════
# 6. RAG pipeline
# ══════════════════════════════════════════════════════════════════════════════

class FinancialLiteracyRAG:
    """
    End-to-end RAG:
        query → embed → language-filtered ChromaDB similarity search
              → reject weak matches and fall back to relevant English guidance
              → return the top matching chunks with their source names
    """

    def __init__(self, docs_dir: str = DOCS_DIR):
        self.collection: chromadb.Collection = get_or_create_collection(docs_dir)

    def answer_with_sources(
        self, question: str, language: str = "English", top_k: int = 5
    ) -> tuple[str, list[str]]:
        """
        Retrieve relevant chunks for *question* in *language*.
        If no relevant chunks exist in that language, try English guidance.
        """
        if not question.strip():
            raise ValueError("Question must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        q_embedding = embed_texts([question])[0]

        # Language filtering keeps English-heavy collections from crowding out
        # matching material in the user's selected language.
        pool_size = max(top_k * 6, 30)
        collection_count = self.collection.count()
        if collection_count == 0:
            return NO_RESULT_MSG.get(language, NO_RESULT_MSG["English"]), []

        def retrieve(language_filter: str) -> list[tuple[str, dict, float]]:
            res = self.collection.query(
                query_embeddings=[q_embedding],
                n_results=min(pool_size, collection_count),
                where={"language": language_filter},
                include=["documents", "metadatas", "distances"],
            )
            documents = res["documents"][0] if res["documents"] else []
            metadatas = res["metadatas"][0] if res["metadatas"] else []
            distances = res["distances"][0] if res["distances"] else []
            matches = list(zip(documents, metadatas, distances))
            if not matches:
                return []

            max_distance = min(
                MAX_RETRIEVAL_DISTANCE,
                matches[0][2] + RETRIEVAL_DISTANCE_MARGIN,
            )
            return [
                (doc, metadata, distance)
                for doc, metadata, distance in matches
                if distance <= max_distance
            ][:top_k]

        chosen = retrieve(language)
        if not chosen and language != "English":
            chosen = retrieve("English")

        if not chosen:
            return NO_RESULT_MSG.get(language, NO_RESULT_MSG["English"]), []

        context = "\n\n".join(doc for doc, _, _ in chosen)
        answer = _generate_answer(question, context, language)
        sources = list(dict.fromkeys(
            metadata["source"]
            for _, metadata, _ in chosen
            if metadata.get("source")
        ))
        return answer, sources

    def answer(self, question: str, language: str = "English", top_k: int = 5) -> str:
        """Return a retrieval answer without source metadata for existing callers."""
        answer, _ = self.answer_with_sources(question, language, top_k)
        return answer

    def reingest(self, docs_dir: str = DOCS_DIR) -> None:
        """Drop all vectors and re-ingest from *docs_dir*."""
        client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        client.delete_collection(COLLECTION_NAME)
        self.collection = ingest_documents(docs_dir)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Built-in seed knowledge base (English + 5 regional languages)
# ══════════════════════════════════════════════════════════════════════════════

def _get_seeded_records() -> list[dict]:
    """Full multilingual seed knowledge so answers come out in the selected language."""

    # ── English ────────────────────────────────────────────────────────────
    english = [
        ("UPI_Safety_Tips_en.txt", "English",
         """UPI Safety Guidelines:
1. Never share your UPI PIN with anyone, including bank officials.
2. Always verify the recipient's UPI ID and amount before confirming. If a transfer goes wrong, contact your UPI app or bank promptly; recovery is not guaranteed.
3. Ignore collect requests from unknown parties. Scammers send fake collect requests.
4. Use the official bank app or BHIM app. Avoid third-party apps of unknown origin.
5. Enable transaction limits and two-factor authentication in your UPI app settings.
6. Ensure your registered mobile number is always linked to your bank account.
7. Report fraud immediately: National Cyber Crime helpline 1930 or cybercrime.gov.in."""),

        ("Online_Scams_en.txt", "English",
         """Common Online Financial Scams in India:
1. OTP Fraud: Fraudsters pose as bank officials and request OTPs. Never share OTPs.
2. KYC Update Scam: Fake messages claim your account will be blocked unless you update KYC.
3. Lottery/Prize Scam: You have won a prize, pay a processing fee first. Legitimate lotteries never ask for upfront fees.
4. Job Offer Scam: Fake jobs that require a registration fee or banking details.
5. Investment Doubling Scam: Promises of 2x or 3x returns in short periods — always a red flag.
6. QR Code Scam: A scammer may ask you to scan a QR code and enter your PIN to send them money; scanning alone does not debit your account.
Red flags: urgency, unsolicited contact, requests for PIN or OTP, upfront processing fees."""),

        ("Budgeting_en.txt", "English",
         """Simple Budgeting — The 50-30-20 Rule:
50% of income goes to Needs: rent, groceries, utilities, loan EMIs.
30% of income goes to Wants: dining out, entertainment, shopping.
20% of income goes to Savings and investments: emergency fund, SIP, PPF.
Practical tips:
1. Track every expense using a notebook or free app.
2. Build an emergency fund covering 3 to 6 months of living expenses before investing.
3. Pay yourself first — transfer savings on salary day before spending.
4. Observe a 24-hour wait rule before purchasing anything above Rs 500.
5. Audit subscriptions every quarter and cancel ones you do not use."""),

        ("Digital_Payments_en.txt", "English",
         """Safe Digital Payment Practices:
1. Use only official apps from RBI-regulated banks and NPCI-authorised platforms.
2. Keep your phone OS and payment apps updated to receive security patches.
3. Use a strong unique password and enable biometric lock on your payment app.
4. Never make payments over public Wi-Fi; use mobile data instead.
5. Review your bank statement weekly for unauthorised transactions.
6. Set a daily transaction limit in your net-banking settings.
7. Register for SMS and email transaction alerts so every debit is notified instantly."""),

        ("Govt_Schemes_en.txt", "English",
         """Key Government Financial Schemes for Indian Citizens:
1. PM Jan Dhan Yojana: Zero-balance savings account with a RuPay debit card and Rs 2 lakh accident insurance cover.
2. Atal Pension Yojana: Government co-contribution pension scheme for unorganised-sector workers aged 18 to 40.
3. PM Suraksha Bima Yojana: Accidental death or disability cover of Rs 2 lakh at just Rs 20 per year.
4. PM Jeevan Jyoti Bima Yojana: Life cover of Rs 2 lakh at Rs 436 per year for bank account holders aged 18 to 50.
5. Sukanya Samriddhi Yojana: High-interest savings scheme for girl children; account can be opened up to age 10.
6. National Savings Certificate: Risk-free 5-year government investment with tax benefits under Section 80C."""),
    ]

    # ── Hindi ──────────────────────────────────────────────────────────────
    hindi = [
        ("UPI_Safety_Tips_hi.txt", "Hindi",
         """UPI सुरक्षा दिशानिर्देश:
1. अपना UPI PIN किसी के साथ साझा न करें, यहां तक कि बैंक अधिकारियों के साथ भी नहीं।
2. पुष्टि करने से पहले प्राप्तकर्ता का UPI ID और राशि जाँचें। गलत भुगतान होने पर तुरंत UPI ऐप या बैंक से संपर्क करें; पैसे वापस मिलने की गारंटी नहीं है।
3. अज्ञात पक्षों से collect request को अनदेखा करें। स्कैमर्स नकली collect request भेजते हैं।
4. आधिकारिक बैंक ऐप या BHIM ऐप का उपयोग करें। अज्ञात मूल के तृतीय-पक्ष ऐप से बचें।
5. अपनी UPI ऐप सेटिंग में लेनदेन सीमाएं और दो-कारक प्रमाणीकरण सक्षम करें।
6. सुनिश्चित करें कि आपका पंजीकृत मोबाइल नंबर हमेशा आपके बैंक खाते से जुड़ा हो।
7. धोखाधड़ी की तुरंत रिपोर्ट करें: राष्ट्रीय साइबर अपराध हेल्पलाइन 1930 या cybercrime.gov.in।"""),

        ("Online_Scams_hi.txt", "Hindi",
         """भारत में सामान्य ऑनलाइन वित्तीय धोखाधड़ी:
1. OTP धोखाधड़ी: धोखेबाज बैंक अधिकारियों के रूप में पेश होते हैं और OTP मांगते हैं। कभी OTP साझा न करें।
2. KYC अपडेट स्कैम: नकली संदेश दावा करते हैं कि आपका खाता ब्लॉक हो जाएगा जब तक आप KYC अपडेट नहीं करते।
3. लॉटरी/पुरस्कार स्कैम: आपने पुरस्कार जीता है, पहले प्रसंस्करण शुल्क का भुगतान करें। वैध लॉटरी अग्रिम शुल्क नहीं लेतीं।
4. नौकरी ऑफर स्कैम: नकली नौकरियां जिनके लिए पंजीकरण शुल्क या बैंकिंग विवरण की आवश्यकता होती है।
5. निवेश दोगुना स्कैम: कम समय में 2x या 3x रिटर्न का वादा — हमेशा एक खतरे का संकेत।
6. QR कोड स्कैम: धोखेबाज QR स्कैन करके PIN डालने को कह सकता है ताकि आप उसे पैसे भेजें; केवल स्कैन करने से पैसे नहीं कटते।
खतरे के संकेत: जल्दबाजी, बिना मांगे संपर्क, PIN या OTP की मांग, अग्रिम शुल्क।"""),

        ("Budgeting_hi.txt", "Hindi",
         """सरल बजट बनाना — 50-30-20 नियम:
आय का 50% जरूरतों पर: किराया, किराने का सामान, उपयोगिताएं, EMI।
आय का 30% इच्छाओं पर: बाहर खाना, मनोरंजन, खरीदारी।
आय का 20% बचत और निवेश पर: आपातकालीन निधि, SIP, PPF।
व्यावहारिक सुझाव:
1. एक नोटबुक या मुफ्त ऐप का उपयोग करके हर खर्च का ट्रैक रखें।
2. निवेश करने से पहले 3 से 6 महीने के जीवन-यापन खर्च को कवर करने वाला आपातकालीन फंड बनाएं।
3. वेतन के दिन बचत राशि पहले ट्रांसफर करें।
4. रुपये 500 से अधिक की कोई भी खरीदारी से पहले 24 घंटे प्रतीक्षा करें।
5. हर तिमाही सदस्यता ऑडिट करें और जो उपयोग नहीं हो रहे उन्हें रद्द करें।"""),

        ("Digital_Payments_hi.txt", "Hindi",
         """सुरक्षित डिजिटल भुगतान प्रथाएं:
1. केवल RBI-विनियमित बैंकों और NPCI-अधिकृत प्लेटफार्मों से आधिकारिक ऐप का उपयोग करें।
2. सुरक्षा पैच प्राप्त करने के लिए अपना फोन OS और भुगतान ऐप अपडेट रखें।
3. एक मजबूत अद्वितीय पासवर्ड और बायोमेट्रिक लॉक सक्षम करें।
4. सार्वजनिक Wi-Fi पर कभी भुगतान न करें; इसके बजाय मोबाइल डेटा का उपयोग करें।
5. अनधिकृत लेनदेन के लिए साप्ताहिक अपना बैंक स्टेटमेंट देखें।
6. अपनी नेट-बैंकिंग सेटिंग में दैनिक लेनदेन सीमा निर्धारित करें।
7. SMS और ईमेल लेनदेन अलर्ट के लिए पंजीकरण करें।"""),

        ("Govt_Schemes_hi.txt", "Hindi",
         """भारतीय नागरिकों के लिए प्रमुख सरकारी वित्तीय योजनाएं:
1. PM जन धन योजना: RuPay डेबिट कार्ड और 2 लाख रुपये दुर्घटना बीमा कवर के साथ जीरो-बैलेंस बचत खाता।
2. अटल पेंशन योजना: 18 से 40 वर्ष आयु के असंगठित क्षेत्र के श्रमिकों के लिए सरकारी सह-योगदान पेंशन योजना।
3. PM सुरक्षा बीमा योजना: केवल 20 रुपये प्रति वर्ष पर 2 लाख रुपये का दुर्घटना मृत्यु/विकलांगता कवर।
4. PM जीवन ज्योति बीमा योजना: 18 से 50 वर्ष आयु के बैंक खाताधारकों के लिए 436 रुपये प्रति वर्ष पर 2 लाख रुपये का जीवन कवर।
5. सुकन्या समृद्धि योजना: बालिकाओं के लिए उच्च-ब्याज बचत योजना; 10 वर्ष तक की आयु तक खाता खोला जा सकता है।
6. राष्ट्रीय बचत पत्र: धारा 80C के तहत कर लाभ के साथ जोखिम-मुक्त 5-वर्षीय सरकारी निवेश।"""),
    ]

    # ── Tamil ──────────────────────────────────────────────────────────────
    tamil = [
        ("UPI_Safety_Tips_ta.txt", "Tamil",
         """UPI பாதுகாப்பு வழிகாட்டுதல்கள்:
1. உங்கள் UPI PIN-ஐ யாரிடமும் பகிர வேண்டாம், வங்கி அதிகாரிகளிடமும் கூட.
2. உறுதிப்படுத்துவதற்கு முன் பெறுநரின் UPI ID மற்றும் தொகையைச் சரிபார்க்கவும். தவறான பரிவர்த்தனை நடந்தால் உடனே UPI செயலி அல்லது வங்கியைத் தொடர்புகொள்ளுங்கள்; பணம் திரும்பும் என உறுதி இல்லை.
3. அறியாத நபர்களிடமிருந்து collect request-களை புறக்கணிக்கவும். மோசடி செய்பவர்கள் போலி collect request அனுப்புவார்கள்.
4. அதிகாரப்பூர்வ வங்கி ஆப் அல்லது BHIM ஆப்பை பயன்படுத்தவும். அறியாத மூன்றாம் தரப்பு ஆப்களை தவிர்க்கவும்.
5. உங்கள் UPI ஆப் அமைப்புகளில் பரிவர்த்தனை வரம்புகளையும் இரு-காரணி அங்கீகாரத்தையும் இயக்கவும்.
6. உங்கள் பதிவுசெய்யப்பட்ட மொபைல் எண் எப்போதும் வங்கி கணக்குடன் இணைக்கப்பட்டிருப்பதை உறுதிப்படுத்தவும்.
7. மோசடியை உடனடியாக புகார் செய்யுங்கள்: தேசிய சைபர் கிரைம் ஹெல்ப்லைன் 1930."""),

        ("Online_Scams_ta.txt", "Tamil",
         """இந்தியாவில் பொதுவான ஆன்லைன் நிதி மோசடிகள்:
1. OTP மோசடி: மோசடி செய்பவர்கள் வங்கி அதிகாரிகளாக நடித்து OTP கேட்கிறார்கள். OTP-ஐ யாரிடமும் பகிர வேண்டாம்.
2. KYC புதுப்பிப்பு மோசடி: உங்கள் கணக்கு தடுக்கப்படும் என்று போலி செய்திகள் வரும். அந்த இணைப்பை கிளிக் செய்யாதீர்கள்.
3. லாட்டரி/பரிசு மோசடி: நீங்கள் பரிசு வென்றீர்கள், முதலில் கட்டணம் செலுத்துங்கள் என்று சொல்வார்கள். சரியான லாட்டரிகள் முன்கூட்டிய கட்டணம் கேட்காது.
4. வேலை வாய்ப்பு மோசடி: பதிவு கட்டணம் அல்லது வங்கி விவரங்கள் தேவைப்படும் போலி வேலைகள்.
5. முதலீட்டு இரட்டிப்பு மோசடி: குறுகிய காலத்தில் 2x அல்லது 3x வருமானம் என்ற வாக்குறுதிகள் — எப்போதும் ஒரு எச்சரிக்கை அறிகுறி.
6. QR குறியீடு மோசடி: பணம் அனுப்ப PIN உள்ளிடச் சொல்லி மோசடிக்காரர் QR குறியீட்டை ஸ்கேன் செய்யச் சொல்லலாம்; ஸ்கேன் செய்வது மட்டும் பணத்தை எடுக்காது.
அபாய அறிகுறிகள்: அவசரம், கோரப்படாத தொடர்பு, PIN அல்லது OTP கோரிக்கைகள்."""),

        ("Budgeting_ta.txt", "Tamil",
         """எளிய பட்ஜெட் திட்டமிடல் — 50-30-20 விதி:
வருமானத்தின் 50% தேவைகளுக்கு: வாடகை, மளிகை, பயன்பாடுகள், EMI.
வருமானத்தின் 30% விருப்பங்களுக்கு: வெளியே சாப்பிடுவது, பொழுதுபோக்கு, ஷாப்பிங்.
வருமானத்தின் 20% சேமிப்பு மற்றும் முதலீட்டிற்கு: அவசரகால நிதி, SIP, PPF.
நடைமுறை குறிப்புகள்:
1. ஒரு நோட்புக் அல்லது இலவச ஆப் மூலம் ஒவ்வொரு செலவையும் கண்காணிக்கவும்.
2. முதலீடு செய்வதற்கு முன் 3 முதல் 6 மாத வாழ்க்கை செலவுகளை ஈடு செய்யும் அவசரகால நிதியை உருவாக்கவும்.
3. சம்பளம் கிடைத்த நாளே சேமிப்புத் தொகையை முதலில் மாற்றவும்.
4. ரூ 500-க்கு மேல் எந்த வாங்குதலுக்கும் முன்பு 24 மணி நேரம் காத்திருக்கவும்.
5. ஒவ்வொரு காலாண்டிலும் சந்தாக்களை ஆய்வு செய்து, பயன்படுத்தாதவற்றை ரத்து செய்யவும்."""),

        ("Govt_Schemes_ta.txt", "Tamil",
         """இந்திய குடிமக்களுக்கான முக்கிய அரசு நிதி திட்டங்கள்:
1. PM ஜன் தன் யோஜனா: RuPay டெபிட் கார்டு மற்றும் ரூ 2 லட்சம் விபத்து காப்பீட்டுடன் ஜீரோ-பேலன்ஸ் சேமிப்பு கணக்கு.
2. அடல் பென்ஷன் யோஜனா: 18 முதல் 40 வயது அமைப்பு சாரா துறை தொழிலாளர்களுக்கான அரசு கூட்டுப் பங்களிப்பு பென்ஷன் திட்டம்.
3. PM சுரக்ஷா பீமா யோஜனா: வருடம் வெறும் ரூ 20-க்கு ரூ 2 லட்சம் விபத்து மரண/ஊனம் காப்பீடு.
4. PM ஜீவன் ஜ்யோதி பீமா யோஜனா: 18 முதல் 50 வயது வங்கி கணக்துதாரர்களுக்கு வருடம் ரூ 436-க்கு ரூ 2 லட்சம் உயிர் காப்பீடு.
5. சுகன்யா சம்ருத்தி யோஜனா: பெண் குழந்தைகளுக்கான உயர் வட்டி சேமிப்பு திட்டம்; 10 வயது வரை கணக்கு திறக்கலாம்.
6. தேசிய சேமிப்பு சான்றிதழ்: பிரிவு 80C-ன் கீழ் வரி சலுகைகளுடன் ஆபத்து-இல்லாத 5-ஆண்டு அரசு முதலீடு."""),
    ]

    # ── Telugu ─────────────────────────────────────────────────────────────
    telugu = [
        ("UPI_Safety_Tips_te.txt", "Telugu",
         """UPI భద్రతా మార్గదర్శకాలు:
1. మీ UPI PIN ను ఎవరితోనూ పంచుకోవద్దు, బ్యాంక్ అధికారులతో కూడా వద్దు.
2. నిర్ధారించే ముందు గ్రహీత UPI ID మరియు మొత్తాన్ని తనిఖీ చేయండి. తప్పు లావాదేవీ జరిగితే వెంటనే UPI యాప్ లేదా బ్యాంకును సంప్రదించండి; డబ్బు తిరిగి వస్తుందని హామీ లేదు.
3. తెలియని వ్యక్తుల నుండి collect request లను నిర్లక్ష్యం చేయండి. స్కామర్లు నకిలీ collect request లు పంపుతారు.
4. అధికారిక బ్యాంక్ యాప్ లేదా BHIM యాప్ ని ఉపయోగించండి. తెలియని మూలం యొక్క మూడో పక్ష యాప్ లను నివారించండి.
5. మీ UPI యాప్ సెట్టింగులలో లావాదేవీ పరిమితులు మరియు రెండు-కారక ధృవీకరణను ప్రారంభించండి.
6. మీ నమోదిత మొబైల్ నంబర్ ఎల్లప్పుడూ మీ బ్యాంక్ ఖాతాకు అనుసంధానించబడి ఉందని నిర్ధారించుకోండి.
7. మోసాన్ని వెంటనే రిపోర్ట్ చేయండి: జాతీయ సైబర్ క్రైమ్ హెల్ప్‌లైన్ 1930."""),

        ("Online_Scams_te.txt", "Telugu",
         """భారతదేశంలో సాధారణ ఆన్‌లైన్ ఆర్థిక మోసాలు:
1. OTP మోసం: మోసగాళ్ళు బ్యాంక్ అధికారులుగా నటించి OTP లు అడుగుతారు. OTP లను ఎప్పుడూ పంచుకోవద్దు.
2. KYC నవీకరణ స్కామ్: మీ ఖాతా బ్లాక్ అవుతుందని నకిలీ సందేశాలు వస్తాయి. ఆ లింక్ ను క్లిక్ చేయవద్దు.
3. లాటరీ/బహుమతి స్కామ్: మీరు బహుమతి గెలిచారు, ముందుగా ప్రాసెసింగ్ ఫీజు చెల్లించండి అని చెప్తారు. చట్టబద్ధమైన లాటరీలు ముందుగా రుసుము అడగవు.
4. ఉద్యోగ అందింపు స్కామ్: నమోదు ఫీజు లేదా బ్యాంకింగ్ వివరాలు అవసరమయ్యే నకిలీ ఉద్యోగాలు.
5. పెట్టుబడి రెట్టింపు స్కామ్: తక్కువ కాలంలో 2x లేదా 3x రాబడుల వాగ్దానాలు — ఎల్లప్పుడూ ప్రమాద సంకేతం.
6. QR కోడ్ స్కామ్: డబ్బు పంపేందుకు PIN నమోదు చేయమని మోసగాడు QR కోడ్ స్కాన్ చేయమని అడగవచ్చు; స్కాన్ చేయడం మాత్రమే ఖాతా నుండి డబ్బు తీసివేయదు.
హెచ్చరిక సంకేతాలు: అత్యవసరత, అభ్యర్థించని పరిచయం, PIN లేదా OTP అభ్యర్థనలు, ముందుగా రుసుము."""),

        ("Budgeting_te.txt", "Telugu",
         """సులభమైన బడ్జెట్ రూపొందించడం — 50-30-20 నియమం:
ఆదాయంలో 50% అవసరాలకు: అద్దె, కిరాణా, యుటిలిటీలు, EMI.
ఆదాయంలో 30% కోరికలకు: బయట తినడం, వినోదం, షాపింగ్.
ఆదాయంలో 20% పొదుపు మరియు పెట్టుబడులకు: అత్యవసర నిధి, SIP, PPF.
ఆచరణాత్మక చిట్కాలు:
1. ఒక నోట్‌బుక్ లేదా ఉచిత యాప్ ఉపయోగించి ప్రతి ఖర్చును ట్రాక్ చేయండి.
2. పెట్టుబడి పెట్టే ముందు 3 నుండి 6 నెలల జీవన ఖర్చులను కవర్ చేసే అత్యవసర నిధిని నిర్మించండి.
3. జీతం వచ్చిన రోజే పొదుపు మొత్తాన్ని ముందుగా ట్రాన్స్‌ఫర్ చేయండి.
4. రూ 500 కంటే ఎక్కువ ఏదైనా కొనే ముందు 24 గంటలు వేచి ఉండండి.
5. ప్రతి త్రైమాసికంలో సబ్‌స్క్రిప్షన్‌లను ఆడిట్ చేసి, ఉపయోగించనివాటిని రద్దు చేయండి."""),

        ("Govt_Schemes_te.txt", "Telugu",
         """భారతీయ పౌరులకు ప్రధాన ప్రభుత్వ ఆర్థిక పథకాలు:
1. PM జన్ ధన్ యోజన: RuPay డెబిట్ కార్డ్ మరియు రూ 2 లక్షల ప్రమాద బీమా కవరేజ్‌తో జీరో-బ్యాలెన్స్ పొదుపు ఖాతా.
2. అటల్ పెన్షన్ యోజన: 18 నుండి 40 సంవత్సరాల వయస్సు అసంఘటిత రంగ కార్మికులకు ప్రభుత్వ సహ-సహకార పెన్షన్ పథకం.
3. PM సురక్ష బీమా యోజన: సంవత్సరానికి కేవలం రూ 20 కి రూ 2 లక్షల ప్రమాద మరణం/వైకల్యం కవరేజ్.
4. PM జీవన్ జ్యోతి బీమా యోజన: 18 నుండి 50 సంవత్సరాల వయస్సు బ్యాంక్ ఖాతాదారులకు సంవత్సరానికి రూ 436 కి రూ 2 లక్షల జీవిత కవరేజ్.
5. సుకన్య సమృద్ధి యోజన: బాలికల కోసం అధిక వడ్డీ పొదుపు పథకం; 10 సంవత్సరాల వయస్సు వరకు ఖాతా తెరవొచ్చు.
6. జాతీయ పొదుపు పత్రం: సెక్షన్ 80C కింద పన్ను ప్రయోజనాలతో రిస్క్-ఫ్రీ 5-సంవత్సరాల ప్రభుత్వ పెట్టుబడి."""),
    ]

    # ── Bengali ────────────────────────────────────────────────────────────
    bengali = [
        ("UPI_Safety_Tips_bn.txt", "Bengali",
         """UPI নিরাপত্তা নির্দেশিকা:
1. আপনার UPI PIN কারো সাথে শেয়ার করবেন না, এমনকি ব্যাংক কর্মকর্তাদের সাথেও নয়।
2. নিশ্চিত করার আগে প্রাপকের UPI ID ও টাকার পরিমাণ যাচাই করুন। ভুল লেনদেন হলে দ্রুত UPI অ্যাপ বা ব্যাংকের সঙ্গে যোগাযোগ করুন; টাকা ফেরত নিশ্চিত নয়।
3. অজানা পক্ষের collect request উপেক্ষা করুন। স্ক্যামাররা নকল collect request পাঠায়।
4. অফিসিয়াল ব্যাংক অ্যাপ বা BHIM অ্যাপ ব্যবহার করুন। অজানা উৎসের তৃতীয় পক্ষের অ্যাপ এড়িয়ে চলুন।
5. আপনার UPI অ্যাপ সেটিংসে লেনদেনের সীমা এবং দুই-ফ্যাক্টর প্রমাণীকরণ সক্ষম করুন।
6. নিশ্চিত করুন যে আপনার নিবন্ধিত মোবাইল নম্বর সর্বদা আপনার ব্যাংক অ্যাকাউন্টের সাথে সংযুক্ত।
7. প্রতারণা অবিলম্বে রিপোর্ট করুন: জাতীয় সাইবার ক্রাইম হেল্পলাইন 1930।"""),

        ("Budgeting_bn.txt", "Bengali",
         """সহজ বাজেট পরিকল্পনা — ৫০-৩০-২০ নিয়ম:
আয়ের ৫০% প্রয়োজনে: ভাড়া, মুদিখানা, ইউটিলিটি, EMI।
আয়ের ৩০% ইচ্ছায়: বাইরে খাওয়া, বিনোদন, কেনাকাটা।
আয়ের ২০% সঞ্চয় ও বিনিয়োগে: জরুরি তহবিল, SIP, PPF।
ব্যবহারিক টিপস:
1. একটি নোটবুক বা বিনামূল্যের অ্যাপ ব্যবহার করে প্রতিটি ব্যয় ট্র্যাক করুন।
2. বিনিয়োগের আগে ৩ থেকে ৬ মাসের জীবন যাত্রার খরচ মেটানো জরুরি তহবিল তৈরি করুন।
3. বেতনের দিনই সঞ্চয়ের টাকা আগে স্থানান্তর করুন।
4. ৫০০ টাকার বেশি কিছু কেনার আগে ২৪ ঘন্টা অপেক্ষা করুন।
5. প্রতি কোয়ার্টারে সাবস্ক্রিপশন অডিট করুন এবং ব্যবহার না করা গুলো বাতিল করুন।"""),

        ("Govt_Schemes_bn.txt", "Bengali",
         """ভারতীয় নাগরিকদের জন্য প্রধান সরকারি আর্থিক প্রকল্প:
1. PM জন ধন যোজনা: RuPay ডেবিট কার্ড এবং ২ লক্ষ টাকার দুর্ঘটনা বীমা কভারেজ সহ জিরো-ব্যালেন্স সঞ্চয় অ্যাকাউন্ট।
2. অটল পেনশন যোজনা: ১৮ থেকে ৪০ বছর বয়সী অসংগঠিত খাতের শ্রমিকদের জন্য সরকারি সহ-অবদান পেনশন প্রকল্প।
3. PM সুরক্ষা বিমা যোজনা: মাত্র ২০ টাকা প্রতি বছরে ২ লক্ষ টাকার দুর্ঘটনা মৃত্যু/প্রতিবন্ধী কভারেজ।
4. PM জীবন জ্যোতি বিমা যোজনা: ১৮ থেকে ৫০ বছর বয়সী ব্যাংক অ্যাকাউন্ট ধারকদের জন্য ৪৩৬ টাকা প্রতি বছরে ২ লক্ষ টাকার জীবন কভারেজ।
5. সুকন্যা সমৃদ্ধি যোজনা: মেয়ে শিশুদের জন্য উচ্চ-সুদের সঞ্চয় প্রকল্প; ১০ বছর বয়স পর্যন্ত অ্যাকাউন্ট খোলা যাবে।
6. জাতীয় সঞ্চয় পত্র: ধারা ৮০সি-এর অধীনে কর সুবিধা সহ ঝুঁকিমুক্ত ৫-বছরের সরকারি বিনিয়োগ।"""),
    ]

    # ── Marathi ────────────────────────────────────────────────────────────
    marathi = [
        ("UPI_Safety_Tips_mr.txt", "Marathi",
         """UPI सुरक्षा मार्गदर्शक तत्त्वे:
1. आपला UPI PIN कोणाशीही शेअर करू नका, बँक अधिकाऱ्यांशीही नाही.
2. पुष्टी करण्यापूर्वी प्राप्तकर्त्याचा UPI ID आणि रक्कम तपासा. चुकीचा व्यवहार झाल्यास त्वरित UPI अॅप किंवा बँकेशी संपर्क साधा; पैसे परत मिळण्याची खात्री नाही.
3. अज्ञात पक्षांच्या collect request दुर्लक्ष करा. फसवे लोक बनावट collect request पाठवतात.
4. अधिकृत बँक अ‍ॅप किंवा BHIM अ‍ॅप वापरा. अज्ञात मूळाच्या तृतीय-पक्ष अ‍ॅप्स टाळा.
5. आपल्या UPI अ‍ॅप सेटिंग्जमध्ये व्यवहार मर्यादा आणि दोन-घटक प्रमाणीकरण सक्षम करा.
6. आपला नोंदणीकृत मोबाइल नंबर नेहमी आपल्या बँक खात्याशी जोडलेला असल्याची खात्री करा.
7. फसवणूकीची ताबडतोब तक्रार करा: राष्ट्रीय सायबर क्राइम हेल्पलाइन 1930."""),

        ("Budgeting_mr.txt", "Marathi",
         """सोपे बजेट नियोजन — ५०-३०-२० नियम:
उत्पन्नाचे ५०% गरजांसाठी: भाडे, किराणा, उपयुक्तता, EMI.
उत्पन्नाचे ३०% इच्छांसाठी: बाहेर जेवण, मनोरंजन, खरेदी.
उत्पन्नाचे २०% बचत आणि गुंतवणुकीसाठी: आपत्कालीन निधी, SIP, PPF.
व्यावहारिक टिप्स:
1. एक नोटबुक किंवा मोफत अ‍ॅप वापरून प्रत्येक खर्चाचा मागोवा घ्या.
2. गुंतवणूक करण्यापूर्वी ३ ते ६ महिन्यांचा राहणीमान खर्च भागवणारा आपत्कालीन निधी तयार करा.
3. पगाराच्या दिवशी बचत रक्कम आधी हस्तांतरित करा.
4. रु ५०० पेक्षा जास्त कोणतीही खरेदी करण्यापूर्वी २४ तास प्रतीक्षा करा.
5. दर तिमाहीत सदस्यता ऑडिट करा आणि न वापरलेल्या रद्द करा."""),

        ("Govt_Schemes_mr.txt", "Marathi",
         """भारतीय नागरिकांसाठी प्रमुख सरकारी आर्थिक योजना:
1. PM जन धन योजना: RuPay डेबिट कार्ड आणि रु २ लाख अपघात विमा संरक्षणासह शून्य-शिल्लक बचत खाते.
2. अटल पेन्शन योजना: १८ ते ४० वर्षे वयोगटातील असंघटित क्षेत्रातील कामगारांसाठी सरकारी सह-योगदान पेन्शन योजना.
3. PM सुरक्षा बिमा योजना: वर्षाला फक्त रु २० मध्ये रु २ लाख अपघाती मृत्यू/अपंगत्व संरक्षण.
4. PM जीवन ज्योती बिमा योजना: १८ ते ५० वर्षे वयोगटातील बँक खातेधारकांसाठी वर्षाला रु ४३६ मध्ये रु २ लाख जीवन संरक्षण.
5. सुकन्या समृद्धी योजना: मुलींसाठी उच्च-व्याज बचत योजना; १० वर्षे वयापर्यंत खाते उघडता येते.
6. राष्ट्रीय बचत पत्र: कलम ८०C अंतर्गत कर लाभांसह जोखीममुक्त ५-वर्षांची सरकारी गुंतवणूक."""),
    ]

    all_entries = english + hindi + tamil + telugu + bengali + marathi
    records = [
        {"text": text, "source": source, "language": lang}
        for source, lang, text in all_entries
    ]

    additional_guidance = [
        ("Online_Scams_bn.txt", "Bengali", """সাধারণ অনলাইন আর্থিক প্রতারণা:
1. OTP প্রতারণা: ব্যাংক কর্মী পরিচয়ে OTP চাইলে দেবেন না; ব্যাংক কখনও আপনার PIN বা OTP চাইবে না।
2. KYC প্রতারণা: অ্যাকাউন্ট বন্ধের ভয় দেখানো বার্তার লিংকে ক্লিক না করে ব্যাংকের অফিসিয়াল অ্যাপ বা নম্বরে যাচাই করুন।
3. চাকরি বা পুরস্কারের নামে আগে টাকা, ব্যাংকের গোপন তথ্য বা অ্যাপ ইনস্টল করতে বললে সতর্ক থাকুন।
4. অল্প সময়ে নিশ্চিত বেশি লাভের বিনিয়োগ প্রস্তাব সন্দেহজনক।
5. অচেনা collect অনুরোধ গ্রহণ করবেন না; টাকা পেতে UPI PIN লাগে না।
জরুরি চাপ, অচেনা যোগাযোগ এবং PIN বা OTP চাওয়া প্রতারণার লক্ষণ।"""),
        ("Online_Scams_mr.txt", "Marathi", """सामान्य ऑनलाइन आर्थिक फसवणूक:
1. OTP फसवणूक: बँक कर्मचारी असल्याचे सांगून OTP मागितल्यास देऊ नका; बँक तुमचा PIN किंवा OTP कधीही मागत नाही.
2. KYC फसवणूक: खाते बंद होईल अशा संदेशातील लिंक उघडू नका; बँकेच्या अधिकृत अॅप किंवा क्रमांकावर तपासा.
3. नोकरी किंवा बक्षिसासाठी आधी पैसे, गोपनीय बँक माहिती किंवा अॅप इन्स्टॉल करण्यास सांगितल्यास सावध रहा.
4. कमी वेळात खात्रीशीर मोठा नफा देण्याचे गुंतवणूक आश्वासन संशयास्पद आहे.
5. अनोळखी collect विनंती स्वीकारू नका; पैसे मिळवण्यासाठी UPI PIN लागत नाही.
घाईचा दबाव, अनोळखी संपर्क आणि PIN किंवा OTP मागणे ही फसवणुकीची लक्षणे आहेत."""),
        ("Digital_Payments_ta.txt", "Tamil", """பாதுகாப்பான டிஜிட்டல் பணப்பரிவர்த்தனை:
1. வங்கியின் அதிகாரப்பூர்வ செயலி அல்லது அங்கீகரிக்கப்பட்ட தளங்களை மட்டும் பயன்படுத்துங்கள்.
2. கைபேசி மற்றும் செயலிகளைப் புதுப்பித்து, வலுவான திரைப் பூட்டு மற்றும் செயலிப் பூட்டை அமைக்கவும்.
3. PIN அல்லது OTP-ஐ யாரிடமும் பகிராதீர்கள்; பணம் பெற PIN தேவையில்லை.
4. பணம் செலுத்துவதற்கு முன் பெறுநர் மற்றும் தொகையைச் சரிபார்க்கவும்.
5. வங்கி அறிக்கையைத் தொடர்ந்து சரிபார்த்து, பரிவர்த்தனை அறிவிப்புகளை இயக்கவும்.
6. சந்தேகமான பரிவர்த்தனை இருந்தால் உடனே வங்கியைத் தொடர்புகொண்டு புகாரளிக்கவும்."""),
        ("Digital_Payments_te.txt", "Telugu", """సురక్షిత డిజిటల్ చెల్లింపులు:
1. బ్యాంకు అధికారిక యాప్‌లు లేదా అధీకృత ప్లాట్‌ఫారమ్‌లనే ఉపయోగించండి.
2. ఫోన్ మరియు యాప్‌లను నవీకరించండి; బలమైన స్క్రీన్ లాక్, యాప్ లాక్ పెట్టండి.
3. PIN లేదా OTP ఎవరితోనూ పంచుకోవద్దు; డబ్బు స్వీకరించడానికి PIN అవసరం లేదు.
4. చెల్లించే ముందు గ్రహీత మరియు మొత్తాన్ని నిర్ధారించండి.
5. బ్యాంక్ స్టేట్‌మెంట్‌ను తరచూ పరిశీలించి, లావాదేవీ అలర్ట్‌లను ఆన్ చేయండి.
6. అనుమానాస్పద చెల్లింపు కనిపిస్తే వెంటనే బ్యాంకును సంప్రదించి నివేదించండి."""),
        ("Digital_Payments_bn.txt", "Bengali", """নিরাপদ ডিজিটাল লেনদেন:
1. শুধু ব্যাংকের অফিসিয়াল অ্যাপ বা অনুমোদিত প্ল্যাটফর্ম ব্যবহার করুন।
2. ফোন ও অ্যাপ আপডেট রাখুন; শক্তিশালী স্ক্রিন লক এবং অ্যাপ লক ব্যবহার করুন।
3. PIN বা OTP কাউকে বলবেন না; টাকা পেতে PIN লাগে না।
4. পেমেন্টের আগে প্রাপক ও টাকার পরিমাণ যাচাই করুন।
5. ব্যাংক স্টেটমেন্ট নিয়মিত দেখুন এবং লেনদেনের নোটিফিকেশন চালু রাখুন।
6. সন্দেহজনক লেনদেন দেখলে দ্রুত ব্যাংকের সঙ্গে যোগাযোগ করে রিপোর্ট করুন।"""),
        ("Digital_Payments_mr.txt", "Marathi", """सुरक्षित डिजिटल व्यवहार:
1. फक्त बँकेचे अधिकृत अॅप किंवा अधिकृत प्लॅटफॉर्म वापरा.
2. फोन आणि अॅप अद्ययावत ठेवा; मजबूत स्क्रीन लॉक आणि अॅप लॉक वापरा.
3. PIN किंवा OTP कोणालाही सांगू नका; पैसे मिळवण्यासाठी PIN लागत नाही.
4. पैसे पाठवण्यापूर्वी प्राप्तकर्ता आणि रक्कम तपासा.
5. बँक स्टेटमेंट नियमित पाहा आणि व्यवहार सूचना सुरू ठेवा.
6. संशयास्पद व्यवहार दिसल्यास त्वरित बँकेशी संपर्क साधून तक्रार करा."""),
        ("QR_Reward_Scam_en.txt", "English", """QR Code Prize or Cashback Scam:
Treat unexpected prize or cashback QR codes as suspicious. Do not scan it, approve a collect request, enter your UPI PIN, share an OTP, or install an app to receive money—receiving money never requires your UPI PIN. Verify offers only in your bank or merchant's official app or website. Scanning alone does not debit your account; authorizing a payment with your PIN can. If you already approved a payment or shared credentials, contact your bank immediately and report fraud at 1930 or cybercrime.gov.in."""),
        ("QR_Reward_Scam_hi.txt", "Hindi", """QR कोड से इनाम या कैशबैक का झांसा:
अगर कोई अनजान व्यक्ति इनाम या कैशबैक भेजने के नाम पर QR कोड भेजे, तो सावधान रहें। उसे स्कैन न करें, collect request स्वीकार न करें, UPI PIN न डालें, OTP साझा न करें और पैसे पाने के लिए कोई ऐप इंस्टॉल न करें। पैसे पाने के लिए UPI PIN की ज़रूरत नहीं होती। असली ऑफर की पुष्टि बैंक या व्यापारी के आधिकारिक ऐप या वेबसाइट पर करें। केवल QR स्कैन करने से पैसे नहीं कटते; PIN डालकर भुगतान मंज़ूर करने से पैसे जा सकते हैं। अगर आपने भुगतान मंज़ूर किया या जानकारी साझा की है, तो तुरंत बैंक से संपर्क करें और 1930 या cybercrime.gov.in पर रिपोर्ट करें।"""),
        ("QR_Reward_Scam_ta.txt", "Tamil", """QR குறியீடு பரிசு அல்லது கேஷ்பேக் மோசடி:
பரிசு அல்லது கேஷ்பேக் தருவதாக அறிமுகமில்லாதவர் QR அனுப்பினால் நம்பாதீர்கள். ஸ்கேன் செய்யாதீர்கள்; collect கோரிக்கையை ஏற்காதீர்கள்; பணம் பெற UPI PIN தேவையில்லை—PIN அல்லது OTP பகிராதீர்கள், செயலி நிறுவாதீர்கள். சலுகையை வங்கி அல்லது வணிகரின் அதிகாரப்பூர்வ செயலியில் சரிபார்க்கவும். ஸ்கேன் மட்டும் பணத்தைக் கழிக்காது; PIN உள்ளிட்டு பணம் செலுத்த ஒப்புதல் அளித்தால் பணம் செல்லலாம். ஒப்புதல் அளித்திருந்தால் உடனே வங்கியைத் தொடர்புகொண்டு 1930 அல்லது cybercrime.gov.in-ல் புகார் செய்யுங்கள்."""),
        ("QR_Reward_Scam_te.txt", "Telugu", """QR కోడ్ బహుమతి లేదా క్యాష్‌బ్యాక్ మోసం:
బహుమతి లేదా క్యాష్‌బ్యాక్ పంపుతామని తెలియని వ్యక్తి QR కోడ్ పంపితే అనుమానించండి. దాన్ని స్కాన్ చేయవద్దు, collect అభ్యర్థనను ఆమోదించవద్దు, UPI PIN నమోదు చేయవద్దు, OTP పంచుకోవద్దు, డబ్బు పొందడానికి ఏ యాప్‌నూ ఇన్‌స్టాల్ చేయవద్దు. డబ్బు స్వీకరించడానికి UPI PIN అవసరం లేదు. నిజమైన ఆఫర్‌ను బ్యాంక్ లేదా వ్యాపారి అధికారిక యాప్ లేదా వెబ్‌సైట్‌లో తనిఖీ చేయండి. QR కోడ్‌ను స్కాన్ చేయడం మాత్రమే ఖాతా నుంచి డబ్బు తీసివేయదు; PINతో చెల్లింపును ఆమోదిస్తే డబ్బు వెళ్లవచ్చు. మీరు ఇప్పటికే ఆమోదించినట్లయితే వెంటనే బ్యాంకును సంప్రదించి 1930 లేదా cybercrime.gov.inలో నివేదించండి."""),
        ("QR_Reward_Scam_bn.txt", "Bengali", """QR কোডে পুরস্কার বা ক্যাশব্যাকের প্রতারণা:
অচেনা কেউ পুরস্কার বা ক্যাশব্যাক পাঠানোর কথা বলে QR কোড পাঠালে সন্দেহ করুন। সেটি স্ক্যান করবেন না, collect অনুরোধ গ্রহণ করবেন না, UPI PIN দেবেন না, OTP ভাগ করবেন না এবং টাকা পেতে কোনো অ্যাপ ইনস্টল করবেন না। টাকা পেতে UPI PIN লাগে না। আসল অফার ব্যাংক বা ব্যবসায়ীর অফিসিয়াল অ্যাপ বা ওয়েবসাইটে যাচাই করুন। শুধু QR কোড স্ক্যান করলে টাকা কাটে না; PIN দিয়ে পেমেন্ট অনুমোদন করলে টাকা পাঠানো হতে পারে। ইতিমধ্যে অনুমোদন করে থাকলে দ্রুত ব্যাংকের সঙ্গে যোগাযোগ করুন এবং 1930 বা cybercrime.gov.in-এ রিপোর্ট করুন।"""),
        ("QR_Reward_Scam_mr.txt", "Marathi", """QR कोडद्वारे बक्षीस किंवा कॅशबॅक फसवणूक:
बक्षीस किंवा कॅशबॅक पाठवण्याचे सांगून अनोळखी व्यक्तीने QR कोड पाठवला तर सावध राहा. तो स्कॅन करू नका, collect विनंती स्वीकारू नका, UPI PIN टाकू नका, OTP शेअर करू नका आणि पैसे मिळवण्यासाठी कोणतेही अॅप इन्स्टॉल करू नका. पैसे मिळवण्यासाठी UPI PIN लागत नाही. खरी ऑफर बँक किंवा व्यापाऱ्याच्या अधिकृत अॅप किंवा वेबसाइटवर तपासा. फक्त QR कोड स्कॅन केल्याने पैसे कापले जात नाहीत; PIN वापरून पेमेंट मंजूर केल्यास पैसे जाऊ शकतात. आधीच पेमेंट मंजूर केले असल्यास लगेच बँकेशी संपर्क साधा आणि 1930 किंवा cybercrime.gov.in वर तक्रार करा."""),
        ("UPI_Transfer_en.txt", "English", """How to Send Money Using UPI:
1. Open your bank's official UPI app and choose Send or Pay.
2. Select the recipient using their verified UPI ID, phone number, or QR code.
3. Check the recipient name and amount carefully before proceeding.
4. Enter your UPI PIN only in the app to authorize the payment. You do not need a PIN to receive money.
5. Check the confirmation screen and transaction history. If a payment is pending, check its status before trying again.
Never approve an unexpected collect request or share your PIN, OTP, or screen with anyone."""),
        ("Loan_Interest_en.txt", "English", """Understanding Loan Interest Rates:
There is no single safe or fair interest rate for every borrower. Compare offers from regulated banks and RBI-registered lenders, and compare the annual percentage rate (APR), total amount repayable, fees, late charges, and whether the rate is fixed or floating.
Ask for the loan's Key Facts Statement and repayment schedule before signing. Check that the EMI fits your budget, read the agreement, and avoid lenders who pressure you to decide immediately or ask for an upfront fee to release a loan. This is general education, not personalized financial advice."""),

        ("UPI_Transfer_hi.txt", "Hindi", """UPI से पैसे कैसे भेजें:
1. अपने बैंक के आधिकारिक UPI ऐप में Send या Pay चुनें।
2. प्राप्तकर्ता का UPI ID, फ़ोन नंबर या QR कोड चुनें और नाम की पुष्टि करें।
3. राशि ध्यान से जाँचें। भुगतान की अनुमति देने के लिए अपना UPI PIN केवल ऐप में डालें।
4. पैसे पाने के लिए PIN की ज़रूरत नहीं होती। अनजान collect request स्वीकार न करें और PIN, OTP या स्क्रीन साझा न करें।
5. भुगतान लंबित हो तो दोबारा भेजने से पहले ऐप में स्थिति जाँचें।"""),
        ("Loan_Interest_hi.txt", "Hindi", """ऋण की ब्याज दर समझना:
हर व्यक्ति और ऋण के लिए एक ही सुरक्षित ब्याज दर नहीं होती। RBI-विनियमित बैंक और RBI-पंजीकृत ऋणदाताओं के प्रस्तावों की वार्षिक प्रतिशत दर (APR), कुल चुकाई जाने वाली राशि, शुल्क, विलंब शुल्क और स्थिर या बदलती दर की तुलना करें।
हस्ताक्षर से पहले ऋण का Key Facts Statement और भुगतान अनुसूची माँगें। जाँचें कि EMI आपके बजट में है, समझौता पढ़ें और तुरंत निर्णय या अग्रिम शुल्क का दबाव बनाने वाले ऋणदाताओं से सावधान रहें। यह सामान्य जानकारी है, व्यक्तिगत वित्तीय सलाह नहीं।"""),

        ("UPI_Transfer_ta.txt", "Tamil", """UPI மூலம் பணம் அனுப்புவது எப்படி:
1. உங்கள் வங்கியின் அதிகாரப்பூர்வ UPI செயலியில் Send அல்லது Pay என்பதைத் தேர்ந்தெடுக்கவும்.
2. பெறுநரின் UPI ID, தொலைபேசி எண் அல்லது QR குறியீட்டைத் தேர்ந்து பெயரை உறுதிப்படுத்தவும்.
3. தொகையைச் சரிபார்த்து, கட்டணத்தை அங்கீகரிக்க PIN-ஐ செயலியில் மட்டும் உள்ளிடவும்.
4. பணம் பெற PIN தேவையில்லை. தெரியாத collect கோரிக்கையை ஏற்காதீர்கள்; PIN, OTP அல்லது திரையை யாருடனும் பகிராதீர்கள்.
5. பரிவர்த்தனை நிலுவையில் இருந்தால் மீண்டும் அனுப்புவதற்கு முன் அதன் நிலையைச் சரிபார்க்கவும்."""),
        ("Loan_Interest_ta.txt", "Tamil", """கடன் வட்டி விகிதங்களைப் புரிந்துகொள்வது:
அனைவருக்கும் பொருந்தும் ஒரே பாதுகாப்பான வட்டி விகிதம் இல்லை. RBI ஒழுங்குபடுத்தும் வங்கிகள் மற்றும் RBI-யில் பதிவு செய்த கடனளிப்பவர்களின் ஆண்டு சதவீத விகிதம் (APR), மொத்த திருப்பிச் செலுத்தும் தொகை, கட்டணங்கள், தாமதக் கட்டணங்கள், நிலையான அல்லது மாறும் விகிதம் ஆகியவற்றை ஒப்பிடுங்கள்.
கையொப்பமிடுவதற்கு முன் Key Facts Statement மற்றும் திருப்பிச் செலுத்தும் அட்டவணையைக் கேளுங்கள். EMI உங்கள் வரவு செலவுத் திட்டத்திற்குள் உள்ளதா எனச் சரிபார்த்து, ஒப்பந்தத்தைப் படியுங்கள். இது பொதுவான கல்வித் தகவல்; தனிப்பட்ட நிதி ஆலோசனை அல்ல."""),

        ("UPI_Transfer_te.txt", "Telugu", """UPI ద్వారా డబ్బు ఎలా పంపాలి:
1. మీ బ్యాంక్ అధికారిక UPI యాప్‌లో Send లేదా Pay ఎంచుకోండి.
2. గ్రహీత UPI ID, ఫోన్ నంబర్ లేదా QR కోడ్‌ను ఎంచుకుని పేరును నిర్ధారించండి.
3. మొత్తాన్ని జాగ్రత్తగా తనిఖీ చేసి, చెల్లింపును ఆమోదించడానికి PIN‌ను యాప్‌లో మాత్రమే నమోదు చేయండి.
4. డబ్బు స్వీకరించడానికి PIN అవసరం లేదు. తెలియని collect అభ్యర్థనలను ఆమోదించవద్దు; PIN, OTP లేదా స్క్రీన్‌ను పంచుకోవద్దు.
5. చెల్లింపు పెండింగ్‌లో ఉంటే మళ్లీ పంపే ముందు స్థితిని తనిఖీ చేయండి."""),
        ("Loan_Interest_te.txt", "Telugu", """రుణ వడ్డీ రేట్లను అర్థం చేసుకోవడం:
ప్రతి ఒక్కరికీ సరిపోయే ఒకే సురక్షిత వడ్డీ రేటు లేదు. RBI నియంత్రిత బ్యాంకులు మరియు RBIలో నమోదైన రుణదాతల వార్షిక శాతం రేటు (APR), మొత్తం తిరిగి చెల్లించాల్సిన మొత్తం, ఫీజులు, ఆలస్య రుసుములు, స్థిర లేదా మారే రేటును పోల్చండి.
సంతకం చేసే ముందు రుణానికి సంబంధించిన Key Facts Statement మరియు చెల్లింపు షెడ్యూల్ అడగండి. EMI మీ బడ్జెట్‌కు సరిపోతుందో చూడండి, ఒప్పందాన్ని చదవండి. ఇది సాధారణ విద్యా సమాచారం మాత్రమే; వ్యక్తిగత ఆర్థిక సలహా కాదు."""),

        ("UPI_Transfer_bn.txt", "Bengali", """UPI দিয়ে টাকা পাঠানোর নিয়ম:
1. আপনার ব্যাংকের অফিসিয়াল UPI অ্যাপে Send বা Pay বেছে নিন।
2. প্রাপকের UPI ID, ফোন নম্বর বা QR কোড বেছে নিয়ে নাম যাচাই করুন।
3. টাকার পরিমাণ দেখে পেমেন্ট অনুমোদনের জন্য শুধু অ্যাপেই PIN দিন।
4. টাকা পেতে PIN লাগে না। অচেনা collect অনুরোধ গ্রহণ করবেন না; PIN, OTP বা স্ক্রিন কারও সঙ্গে ভাগ করবেন না।
5. পেমেন্ট আটকে থাকলে আবার পাঠানোর আগে অ্যাপে তার অবস্থা দেখুন।"""),
        ("Loan_Interest_bn.txt", "Bengali", """ঋণের সুদের হার বোঝা:
সবার জন্য একটিই নিরাপদ সুদের হার নেই। RBI-নিয়ন্ত্রিত ব্যাংক ও RBI-তে নিবন্ধিত ঋণদাতার বার্ষিক শতকরা হার (APR), মোট পরিশোধযোগ্য টাকা, ফি, দেরির চার্জ এবং সুদের হার স্থির না পরিবর্তনশীল—তুলনা করুন।
স্বাক্ষরের আগে Key Facts Statement ও কিস্তির সময়সূচি চান। EMI আপনার বাজেটে মানায় কি না দেখুন এবং চুক্তি পড়ুন। এটি সাধারণ শিক্ষামূলক তথ্য, ব্যক্তিগত আর্থিক পরামর্শ নয়।"""),

        ("UPI_Transfer_mr.txt", "Marathi", """UPI द्वारे पैसे कसे पाठवायचे:
1. तुमच्या बँकेच्या अधिकृत UPI अॅपमध्ये Send किंवा Pay निवडा.
2. प्राप्तकर्त्याचा UPI ID, फोन नंबर किंवा QR कोड निवडून नाव तपासा.
3. रक्कम तपासून पेमेंट मंजूर करण्यासाठी PIN फक्त अॅपमध्ये टाका.
4. पैसे मिळवण्यासाठी PIN लागत नाही. अनोळखी collect विनंती स्वीकारू नका; PIN, OTP किंवा स्क्रीन कोणाशीही शेअर करू नका.
5. पेमेंट प्रलंबित असल्यास पुन्हा पाठवण्यापूर्वी अॅपमध्ये स्थिती तपासा."""),
        ("Loan_Interest_mr.txt", "Marathi", """कर्जाचे व्याजदर समजून घ्या:
प्रत्येकासाठी एकच सुरक्षित व्याजदर नसतो. RBI-नियंत्रित बँका आणि RBI-नोंदणीकृत कर्जदात्यांचे वार्षिक टक्केवारी दर (APR), एकूण परतफेड, शुल्क, विलंब शुल्क आणि दर स्थिर आहे की बदलणारा यांची तुलना करा.
स्वाक्षरीपूर्वी Key Facts Statement आणि परतफेडीचे वेळापत्रक मागा. EMI तुमच्या बजेटमध्ये बसते का ते तपासा आणि करार वाचा. ही सामान्य शैक्षणिक माहिती आहे, वैयक्तिक आर्थिक सल्ला नाही."""),
    ]
    records.extend(
        {"text": text, "source": source, "language": lang}
        for source, lang, text in additional_guidance
    )
    return records


def _write_seed_files(docs_path: Path) -> None:
    """Persist seed content as .txt files so future runs skip re-seeding."""
    docs_path.mkdir(parents=True, exist_ok=True)
    for rec in _get_seeded_records():
        dest = docs_path / rec["source"]
        if not dest.exists():
            dest.write_text(rec["text"], encoding="utf-8")
