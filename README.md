# Music RAG Assistant

## 1. Purpose

This project is a local Retrieval-Augmented Generation (RAG) application built as part of a
summer coding program. It allows users to ask natural-language questions about classical
music and receive answers grounded in a curated knowledge base derived from the MusicNet
dataset, rather than relying on the language model's own (unverified) knowledge.

The domain — classical music — was chosen based on personal interest. The main learning goal
of the project was to understand, step by step, how a RAG pipeline is built: from raw data, to
embeddings, to retrieval, to grounded generation, to a usable interface.

For me, success on this project meant genuinely understanding the AI concepts behind RAG —
what an embedding actually is, why cosine similarity is used to compare them, and how
retrieval and generation fit together — rather than just having a working app. Throughout the
project, the goal was to understand *why* each piece of code was written the way it was (for
example, why the closure pattern was needed for Streamlit's caching, or why a similarity
threshold was necessary at all) instead of just copying a working solution.

## 2. How It Works

**Components:**

| Component | Choice |
|---|---|
| Embedding model | `qwen3-embedding-0.6b` (via Microsoft Foundry Local SDK) |
| Chat / generation model | `phi-3.5-mini` (via Microsoft Foundry Local SDK) |
| Storage | SQLite (`ingest.db`), vectors stored as JSON-serialized text |
| Retrieval method | Cosine similarity, top-k with a similarity threshold |
| Frontend | Streamlit |
| Dataset | MusicNet metadata (Kaggle), transformed into ~123 natural-language documents |

**Pipeline:**

1. **Ingestion (`ingest.py`):** Reads the MusicNet metadata CSV, groups rows by
   `composer + catalog_name`, converts each group into a natural-language text document,
   generates an embedding for each document (batched, 10 at a time to avoid timeouts), and
   stores everything in `ingest.db`.
2. **Retrieval:** The user's question is embedded with the same model, then compared against
   every stored document using cosine similarity. The top-k most similar documents are
   selected, and each one is individually checked against a similarity threshold — a chunk is
   only kept if it clears the threshold on its own, not just if the top result does.
3. **Generation:** The surviving chunks are passed to the chat model as context, along with a
   system prompt instructing it to answer only from the provided context and say "I do not
   know the answer" if the information isn't there. If even the best-matching chunk doesn't
   clear the threshold, the chat model isn't called at all — the app returns "I do not know the
   answer" directly.
4. **Interface (`streamlit_app.py`):** A Streamlit chat interface wraps this pipeline. Model
   loading (which can take several minutes on limited hardware) happens once per session,
   cached via `@st.cache_resource`, with step-by-step progress shown to the user instead of an
   unexplained wait.

## 3. Instructions to Run the App

**Prerequisites:**
- Python 3.14 installed
- Dependencies installed (Streamlit, Microsoft Foundry Local SDK, and their requirements)
- `ingest.db` already created by running `ingest.py` at least once (this populates the
  database with embedded MusicNet documents)

**Steps:**

1. Open a terminal in the project folder (`rag_project`).
2. If `ingest.db` does not exist yet, run the ingestion pipeline first:
   ```
   C:\Users\user\AppData\Local\Programs\Python\Python314\python.exe ingest.py
   ```
3. Launch the Streamlit app:
   ```
   C:\Users\user\AppData\Local\Programs\Python\Python314\python.exe -m streamlit run streamlit_app.py
   ```
4. Streamlit will print a local URL (typically `http://localhost:8501`). Open it in a browser
   if it doesn't open automatically.
5. On the first question asked, the app will download/load the embedding and chat models and
   load the documents from `ingest.db` — this can take a few minutes depending on hardware.
   Progress is shown step by step in the chat window. This only happens once per session.
6. Ask a question about classical music in the chat box at the bottom of the page.

*(Note: the full Python path above is used instead of just `python` or `streamlit` because of a
PATH configuration issue on this machine — see Section 4.)*

## 4. Design Decisions and Limitations

### 4.1 Design decisions

- **Closure pattern for Streamlit state:** `generate_answers` and `answer_query` are defined as
  inner functions inside the `@st.cache_resource`-decorated `load_rag()` function. This lets
  them access the loaded models and documents without relying on module-level globals, and
  ensures the expensive model-loading step only runs once per session instead of on every
  Streamlit re-run (Streamlit re-executes the entire script on every user interaction).
- **Per-chunk hallucination guard:** Every chunk that goes into the model's context is
  individually checked against the similarity threshold, rather than only checking the top
  result. This was a deliberate fix after noticing the original version passed all top-k
  chunks to the model regardless of their individual relevance.
- **Batched embedding generation:** Embedding all ~123 documents in a single API call caused
  timeouts; batching in groups of 10 solved this and would be necessary for scaling to larger
  datasets.
- **Idempotent ingestion:** `ingest.py` deletes existing rows before re-inserting, so re-running
  it does not create duplicate entries.

### 4.2 Limitations

**Threshold sensitivity.** A fixed cosine-similarity threshold does not cleanly separate
answerable from unanswerable queries on a small (~123 document) dataset. For example, a query
about Bach — despite Bach being present in the dataset — returned "I do not know the answer"
when the threshold was set to 0.43, because the best matching document only scored 0.4041, even
though the retrieved content was clearly relevant. The threshold was subsequently lowered to
0.40, after which the Bach query returned a correct answer. This suggests a single scalar
threshold is a fragile mechanism on small or unevenly distributed datasets; more robust
approaches (e.g. relative/adaptive thresholds, re-ranking, or a minimum "top-k agreement" rule)
would likely generalize better.

**Query phrasing affects retrieval quality.** The same underlying question about a composer's
works produced noticeably different quality answers depending on how it was phrased. For
example:
- *"mozart bestecisinin eserleri nelerdir"* (Turkish, casual) → returned a rich, multi-item answer
- *"haydn bestecisinin eserleri nelerdir"* (same phrasing, different composer) → returned only
  one matching work
- *"What are the compositions of the composer Mozart?"* (English, well-formed) → returned the
  most detailed and complete answer of the three

This shows the embedding model's similarity scores — and therefore what gets retrieved — are
sensitive to language and phrasing style, not just semantic content. This is a known limitation
of embedding-based retrieval in general rather than a bug specific to this implementation.

**Case sensitivity (identified and fixed).** During testing, questions about the composer
Cambini failed ("I do not know the answer") when written in lowercase (*"cambini bestecisinin
eserleri nelerdir"*), but succeeded when "Cambini" was capitalized or when asked in English.
This indicated the embedding model produced different vectors depending on capitalization,
likely because the stored documents were embedded with their original capitalized text (e.g.
"Cambini's Wind Quintet..."). The fix was to normalize text to lowercase before generating
embeddings, applied consistently on both sides of the comparison: in `ingest.py`, each document
batch is lowercased before being embedded, and in the query pipeline, the user's question is
lowercased before being embedded. `ingest.py` was then re-run to regenerate all document
embeddings under the new, consistent normalization. After this fix, the same question returned
a relevant answer regardless of capitalization.

**Local model latency.** Running both models on CPU with limited RAM means responses can take
multiple minutes, especially on first load ("cold start"). This occasionally surfaces as a
cancelled/timed-out chat completion on the very first request after starting the app, which
typically succeeds on retry. This is a hardware constraint rather than an application bug.

**Environment/PATH issues.** On this Windows setup, the `python`, `py`, and `streamlit`
commands were not reliably found on PATH, requiring the full interpreter path to be used
instead. A corrupted package installation was also encountered and resolved via
`pip install --force-reinstall streamlit`.