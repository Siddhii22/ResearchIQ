# ResearchIQ

ResearchIQ is an evidence-driven research assistant demo. It searches arXiv, retrieves passages from processed papers, and asks a configured language model to draft an answer grounded in those passages. The Streamlit interface displays the answer alongside cited evidence, source metadata, and a link to each arXiv abstract page.

## How It Works

1. FastAPI turns the question into an arXiv metadata search. The API request is paced to no more than one arXiv API request per three seconds per process.
2. Missing search results may be downloaded and processed locally into page-aware chunks. Cached papers are reused.
3. Dense FAISS retrieval and BM25 are combined with reciprocal-rank fusion and a cross-encoder reranker.
4. The question and retrieved passages are sent to the configured LLM provider. The model is instructed to use the evidence as untrusted source material, not as executable instructions.
5. The response includes evidence passages, page/section details, bibliographic metadata, and a canonical arXiv abstract URL.

The optional LangGraph workflow is available through `scripts/run_agentic_research.py`. Evaluation tools and the UI's local benchmark are separate from live query results; the current displayed benchmark is a 12-question, single-paper evaluation and is not a general performance guarantee.

## Setup

Use Python 3.10 or newer. In PowerShell, from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Put credentials only in `.env` or deployment secrets. For Gemini, configure `LLM_PROVIDER=gemini`, `GEMINI_API_KEY`, and optionally `GEMINI_MODEL`. The optional OpenAI-compatible provider uses `LLM_PROVIDER=openai_compatible`, `LLM_API_KEY`, `LLM_API_BASE_URL`, and `LLM_MODEL`. The frontend receives only the API URL; provider keys belong only in the backend environment. Do not commit `.env` or deployment secrets.

## Run Locally

Start the API in one terminal:

```powershell
\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Start Streamlit in another:

```powershell
$env:RESEARCHIQ_API_URL = "http://127.0.0.1:8000"
\.venv\Scripts\python.exe -m streamlit run frontend\app.py
```

The UI is at `http://localhost:8501`; API health is at `http://127.0.0.1:8000/health`. `docker compose up --build` starts both services. Compose exposes the UI on port 8501 and binds the API host port to loopback.

The API can discover and ingest papers during a research request. To ingest a paper manually, first review the paper's current reuse license and arXiv terms, then run:

```powershell
\.venv\Scripts\python.exe scripts\ingest_paper.py 2401.12345
\.venv\Scripts\python.exe scripts\build_index.py
```

The retrieval settings `DENSE_TOP_K`, `BM25_TOP_K`, `RRF_K`, `RERANK_TOP_K`, `FINAL_TOP_K`, and `RERANK_MODEL` are read from environment variables. The embedding and reranker models are downloaded from Hugging Face when first needed.

## Data, Privacy, and Security

- There is no user-file upload feature or API route for arbitrary local paths. Questions are limited to 4,000 characters, request bodies to 16 KiB, and the research endpoint to 10 requests per minute per observed client address in one API process.
- Application code does not intentionally persist research questions or generated answers. Streamlit keeps the current form/result in its session state while the session is active. Hosting, proxy, and provider logs/retention are outside this application's control.
- The question's search terms are sent to arXiv. The full question and retrieved evidence excerpts are sent to the configured LLM provider for generation. Review that provider's terms, data controls, and retention policy before using sensitive queries. The app does not send uploaded documents because uploads are not supported.
- Downloaded PDFs, extracted page text/chunks, and the FAISS metadata index are stored under local `data/` directories. They are excluded from Git and the Docker build context, and no endpoint serves these directories or full PDF files. Retrieved evidence excerpts are returned to the requesting user. Paper caches are shared by the one demo backend process; this is a public-paper cache, not a per-user store.
- The in-memory rate limiter is per process and uses the socket peer address. Behind a reverse proxy, the API may see the proxy address for every visitor; configure a trusted gateway limit for a public deployment. Do not trust arbitrary client-supplied forwarded-IP headers. Multi-replica deployments need a shared limiter.
- Do not expose the API port publicly without a trusted TLS reverse proxy and deployment-level abuse controls. Keep model credentials in backend secrets, use a restricted provider key/quota, and monitor expected usage.

## Paper Rights and Attribution

arXiv metadata and paper content have different terms. arXiv states that descriptive metadata may be used under CC0, while the paper's copyright and submitted license control content reuse. Its API terms say not to store and serve papers unless the rights holder's permission or the paper's license permits it, and ask API clients to make no more than one legacy API request every three seconds. See the [arXiv API terms](https://info.arxiv.org/help/api/tou.html), [license overview](https://info.arxiv.org/help/license/index.html), and [bulk-data guidance](https://info.arxiv.org/help/bulk_data.html).

ResearchIQ links citations to the original arXiv abstract page and does not provide a PDF-download endpoint. It does cache PDFs and extracted text locally for retrieval and returns selected evidence excerpts. A public operator must review the exact license and applicable terms for every paper and determine whether that processing, storage, and excerpt display are permitted. Public accessibility alone is not permission to redistribute. This repository does not include the local paper cache, evaluation data, or FAISS index. The default example, “Attention Is All You Need” by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin, is [arXiv:1706.03762](https://arxiv.org/abs/1706.03762); its current page displays the [arXiv non-exclusive distribution license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/), which is not by itself a redistribution grant to this project.

Answers are AI-generated and may be incomplete, inaccurate, or incorrectly interpret a cited passage. Verify important claims against the original paper. ResearchIQ is a research aid, not a guarantee of factual accuracy or legal compliance.

## Dependencies and Licenses

The runtime includes FastAPI, Uvicorn, HTTPX, python-dotenv, pypdf, NumPy, FAISS CPU, sentence-transformers, `google-genai`, LangGraph, and Streamlit. Tests use pytest. arXiv is the scholarly metadata/content service; Hugging Face hosts the configured models; Gemini is the default generation provider.

The following direct dependency/model license declarations were spot-checked against upstream license files or model cards: FastAPI (MIT), Uvicorn (BSD-3-Clause), pypdf (BSD-style), NumPy (BSD-3-Clause), FAISS (MIT), Streamlit (Apache-2.0), Google Gen AI SDK (Apache-2.0), LangGraph (MIT), `all-MiniLM-L6-v2` (Apache-2.0), and `ms-marco-MiniLM-L-6-v2` (Apache-2.0). Review each project's current license, notices, model-card terms, and all transitive dependencies before redistribution or commercial deployment; this spot check is not a compatibility determination. `requirements.txt` uses version ranges and no lock file is included. A vulnerability database scan was not available in this environment (`pip-audit` is not installed).

No project-level `LICENSE` file is currently included. The rights to reuse this repository's own code have therefore not been granted by a project license; choose a license only after confirming authorship and ownership.

## Evaluation and Tests

Local evaluation data and results are ignored by Git. When available, the UI displays K=5 metrics from the saved 12-question *Attention Is All You Need* evaluation only. To run tests:

```powershell
\.venv\Scripts\python.exe -m pytest -q
```