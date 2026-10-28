# """Streamlit research interface for ResearchIQ."""

# import json
# import math
# import os
# from html import escape
# from pathlib import Path

# import httpx
# import streamlit as st


# API_URL = os.getenv("RESEARCHIQ_API_URL", "http://127.0.0.1:8000").rstrip("/")
# RESEARCH_ENDPOINT = f"{API_URL}/api/v1/research/query"
# API_RESPONSE_TIMEOUT_SECONDS = 120.0
# API_CONNECT_TIMEOUT_SECONDS = 5.0
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# EVALUATION_RESULTS_PATH = PROJECT_ROOT / "data/eval/results/retrieval_evaluation_results.json"
# EVALUATION_DATASET_PATH = PROJECT_ROOT / "data/eval/attention_transformer_eval.json"

# EXAMPLE_QUESTIONS = (
#     (
#         "Retrieval & grounding",
#         "How does retrieval-augmented generation reduce hallucinations in large language models?",
#     ),
#     (
#         "Deep reinforcement learning",
#         "Which techniques improve sample efficiency in deep reinforcement learning?",
#     ),
#     (
#         "Vision architectures",
#         "How do vision transformers compare with convolutional networks for image recognition?",
#     ),
# )

# EVALUATION_METHODS = (
#     ("dense", "Dense retrieval"),
#     ("bm25", "BM25"),
#     ("hybrid_rrf", "Hybrid RRF"),
#     ("hybrid_rrf_reranker", "Hybrid RRF + Cross-Encoder Reranker"),
# )


# def select_example(question: str) -> None:
#     st.session_state["research_question"] = question


# def load_retrieval_benchmark() -> dict[str, object] | None:
#     try:
#         results = json.loads(EVALUATION_RESULTS_PATH.read_text(encoding="utf-8"))
#     except (OSError, json.JSONDecodeError):
#         return None

#     if not isinstance(results, list):
#         return None

#     try:
#         dataset = json.loads(EVALUATION_DATASET_PATH.read_text(encoding="utf-8"))
#     except (OSError, json.JSONDecodeError):
#         return None
#     if not isinstance(dataset, list) or len(dataset) != 12:
#         return None
#     if any(
#         not isinstance(question, dict)
#         or not isinstance(question.get("question"), str)
#         or not question["question"].strip()
#         or not isinstance(question.get("relevant_chunk_ids"), list)
#         or not question["relevant_chunk_ids"]
#         or any(
#             not isinstance(chunk_id, str) or "-chunk-" not in chunk_id
#             for chunk_id in question["relevant_chunk_ids"]
#         )
#         for question in dataset
#     ):
#         return None
#     paper_ids = {
#         chunk_id.rsplit("-chunk-", 1)[0]
#         for question in dataset
#         for chunk_id in question["relevant_chunk_ids"]
#         if isinstance(chunk_id, str) and "-chunk-" in chunk_id
#     }
#     if len(paper_ids) != 1:
#         return None

#     method_labels = dict(EVALUATION_METHODS)
#     results_by_method = {}
#     for result in results:
#         if not isinstance(result, dict) or result.get("K") != 5:
#             continue
#         method = result.get("method")
#         if not isinstance(method, str):
#             return None
#         if method not in method_labels:
#             continue
#         if method in results_by_method:
#             return None
#         try:
#             recall = result["Recall@K"]
#             mrr = result["MRR@K"]
#             latency = result["average_latency"]
#         except (KeyError, TypeError, ValueError):
#             return None
#         raw_metrics = (recall, mrr, latency)
#         if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_metrics):
#             return None
#         try:
#             recall, mrr, latency = (float(value) for value in raw_metrics)
#         except (OverflowError, TypeError, ValueError):
#             return None
#         if not all(math.isfinite(value) for value in (recall, mrr, latency)):
#             return None
#         if not 0 <= recall <= 1 or not 0 <= mrr <= 1 or latency < 0:
#             return None
#         results_by_method[method] = {
#             "Retrieval method": method_labels[method],
#             "Recall@5": recall,
#             "MRR@5": mrr,
#             "Average latency (s)": latency,
#         }

#     if set(results_by_method) != set(method_labels):
#         return None
#     return {
#         "rows": [results_by_method[method] for method, _ in EVALUATION_METHODS],
#         "question_count": len(dataset),
#     }


# def format_latency(seconds: float) -> str:
#     if seconds < 0.001:
#         return f"{seconds:.6f}" if seconds >= 0.000001 else f"{seconds:.2e}"
#     return f"{seconds:.4f}"


# def render_benchmark_table(benchmark: dict[str, object]) -> None:
#     rows = benchmark.get("rows", [])
#     if not isinstance(rows, list) or not rows:
#         return

#     numeric_rows = []
#     for row in rows:
#         if not isinstance(row, dict):
#             continue
#         numeric_rows.append(
#             {
#                 "method": str(row.get("Retrieval method", "Unknown")),
#                 "recall": float(row.get("Recall@5", 0.0)),
#                 "mrr": float(row.get("MRR@5", 0.0)),
#                 "latency": float(row.get("Average latency (s)", 0.0)),
#             }
#         )

#     if not numeric_rows:
#         return

#     best_recall = max(item["recall"] for item in numeric_rows)
#     best_mrr = max(item["mrr"] for item in numeric_rows)
#     lowest_latency = min(item["latency"] for item in numeric_rows)

#     table_rows = []
#     for item in numeric_rows:
#         recall_class = " metric-best-purple" if abs(item["recall"] - best_recall) < 1e-12 else ""
#         mrr_class = " metric-best-teal" if abs(item["mrr"] - best_mrr) < 1e-12 else ""
#         latency_class = " metric-low-latency" if abs(item["latency"] - lowest_latency) < 1e-12 else ""

#         method = escape(item["method"])
#         recall = f"{item['recall']:.3f}"
#         mrr = f"{item['mrr']:.3f}"
#         latency = f"{format_latency(item['latency'])} s"

#         table_rows.append(
#             "<tr>"
#             f"<td>{method}</td>"
#             f"<td class=\"metric-cell{recall_class}\">{recall}</td>"
#             f"<td class=\"metric-cell{mrr_class}\">{mrr}</td>"
#             f"<td class=\"metric-cell{latency_class}\">{latency}</td>"
#             "</tr>"
#         )

#     table_html = "".join(table_rows)
#     st.markdown(
#         f"""
#         <div class="benchmark-shell">
#             <table class="benchmark-table">
#                 <thead>
#                     <tr>
#                         <th>Retrieval method</th>
#                         <th>Recall@5</th>
#                         <th>MRR@5</th>
#                         <th>Average latency</th>
#                     </tr>
#                 </thead>
#                 <tbody>
#                     {table_html}
#                 </tbody>
#             </table>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )


# st.set_page_config(
#     page_title="ResearchIQ | Research Workspace",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )
# st.markdown(
#     """
#     <style>
#     :root {
#         color-scheme: dark;
#         --page: #0B1120;
#         --panel: #111B2E;
#         --panel-raised: #172338;
#         --surface: #0F172A;
#         --text: #F8FAFC;
#         --secondary: #CBD5E1;
#         --muted: #94A3B8;
#         --border: #334155;
#         --primary-accent: #A78BFA;
#         --secondary-accent: #38BDF8;
#         --success-accent: #2DD4BF;
#         --shadow: rgba(2, 6, 23, 0.52);
#     }

#     html, body, [data-testid="stAppViewContainer"], .stApp {
#         background: var(--page);
#         color: var(--text);
#     }

#     [data-testid="stHeader"] {
#         background: rgba(11, 17, 32, 0.88);
#         border-bottom: 1px solid rgba(51, 65, 85, 0.7);
#     }

#     [data-testid="stMainBlockContainer"] {
#         max-width: 1200px;
#         padding: 4.5rem 2.25rem 4rem;
#     }

#     h1, h2, h3, h4, p, label, li, div, span { color: var(--text); }

#     h1, h2, h3, h4 {
#         letter-spacing: -0.02em;
#         color: var(--text);
#     }

#     [data-testid="stMarkdownContainer"] p,
#     [data-testid="stMarkdownContainer"] li,
#     [data-testid="stCaptionContainer"],
#     .stCaption,
#     .section-description,
#     .metadata-label,
#     [data-testid="stHorizontalBlock"] span,
#     [data-testid="stExpander"] .streamlit-expanderHeader {
#         color: var(--secondary);
#     }

#     .brand-bar {
#         display: flex;
#         align-items: center;
#         justify-content: space-between;
#         gap: 1rem;
#         padding-bottom: 1.25rem;
#         border-bottom: 1px solid var(--border);
#     }

#     .brand-lockup {
#         display: flex;
#         align-items: center;
#         gap: 0.85rem;
#     }

#     .brand-mark {
#         display: grid;
#         place-items: center;
#         width: 44px;
#         height: 44px;
#         border: 1px solid rgba(167, 139, 250, 0.42);
#         border-radius: 12px;
#         background: linear-gradient(145deg, rgba(167, 139, 250, 0.12), rgba(56, 189, 248, 0.08));
#         box-shadow: 0 8px 20px rgba(167, 139, 250, 0.12);
#     }

#     .brand-name {
#         color: var(--text);
#         font-size: 1.55rem;
#         line-height: 1.1;
#         font-weight: 800;
#         letter-spacing: -0.04em;
#     }

#     .brand-subtitle {
#         margin-top: 0.14rem;
#         color: var(--secondary);
#         font-size: 0.78rem;
#         line-height: 1.3;
#     }

#     .workspace-badge {
#         display: inline-flex;
#         align-items: center;
#         gap: 0.55rem;
#         padding: 0.45rem 0.75rem;
#         border: 1px solid rgba(167, 139, 250, 0.24);
#         border-radius: 999px;
#         background: rgba(167, 139, 250, 0.06);
#         color: var(--secondary);
#         font-size: 0.66rem;
#         font-weight: 700;
#         letter-spacing: 0.08em;
#         text-transform: uppercase;
#     }

#     .badge-dot {
#         width: 7px;
#         height: 7px;
#         border-radius: 50%;
#         background: var(--primary-accent);
#         box-shadow: 0 0 10px rgba(167, 139, 250, 0.7);
#     }

#     .hero {
#         max-width: 780px;
#         padding: 2rem 0 1.4rem;
#     }

#     .hero-kicker,
#     .section-kicker,
#     .topic-label,
#     .answer-label {
#         color: var(--primary-accent);
#         font-size: 0.7rem;
#         font-weight: 800;
#         letter-spacing: 0.11em;
#         text-transform: uppercase;
#     }

#     .hero h1 {
#         margin: 0.45rem 0 0.7rem;
#         color: var(--text);
#         font-size: clamp(2.5rem, 3.8vw, 3.5rem);
#         line-height: 1.04;
#         font-weight: 800;
#         letter-spacing: -0.05em;
#     }

#     .hero p {
#         max-width: 62ch;
#         margin: 0;
#         color: var(--secondary);
#         font-size: 1rem;
#         line-height: 1.7;
#     }

#     .section-title {
#         margin: 0.2rem 0 0.35rem;
#         font-size: 1.35rem;
#         font-weight: 700;
#         color: var(--text);
#     }

#     .section-description {
#         margin: 0 0 0.9rem;
#         color: var(--secondary);
#         font-size: 0.94rem;
#         line-height: 1.6;
#     }

#     .site-footer {
#         margin-top: 1rem;
#         padding: 0.75rem 0;
#         color: var(--muted);
#         font-size: 0.78rem;
#         text-align: center;
#     }

#     [data-testid="stVerticalBlockBorderWrapper"] > div {
#         border: 1px solid var(--border) !important;
#         border-radius: 14px !important;
#         background: linear-gradient(180deg, rgba(17, 27, 46, 0.96), rgba(17, 27, 46, 0.9)) !important;
#         box-shadow: none;
#         transition: border-color 150ms ease, box-shadow 150ms ease, transform 150ms ease;
#     }

#     [data-testid="stVerticalBlockBorderWrapper"]:hover > div {
#         border-color: rgba(167, 139, 250, 0.68) !important;
#         box-shadow: 0 12px 28px rgba(15, 23, 42, 0.36);
#     }

#     .topic-label {
#         margin-bottom: 0.5rem;
#         color: var(--secondary-accent);
#         font-size: 0.68rem;
#     }

#     .example-question {
#         min-height: 4.5rem;
#         color: var(--text);
#         font-size: 0.95rem;
#         line-height: 1.6;
#     }

#     [data-testid="stTextArea"] textarea {
#         min-height: 136px;
#         border: 1px solid var(--border);
#         border-radius: 10px;
#         background: rgba(11, 17, 32, 0.7);
#         color: var(--text);
#         line-height: 1.6;
#         resize: vertical;
#     }

#     [data-testid="stTextArea"] textarea:focus {
#         border-color: rgba(167, 139, 250, 0.9);
#         box-shadow: 0 0 0 1px rgba(167, 139, 250, 0.28);
#     }

#     [data-testid="stTextArea"] textarea::placeholder {
#         color: var(--muted);
#     }

#     [data-testid="stButton"] > button {
#         min-height: 2.7rem;
#         border: 1px solid rgba(167, 139, 250, 0.25);
#         border-radius: 10px;
#         background: rgba(23, 35, 56, 0.9);
#         color: var(--text);
#         font-weight: 700;
#         transition: background 150ms ease, border-color 150ms ease, transform 150ms ease, box-shadow 150ms ease;
#     }

#     [data-testid="stButton"] > button:hover {
#         border-color: rgba(167, 139, 250, 0.55);
#         background: rgba(34, 46, 67, 0.96);
#         color: var(--text);
#         transform: translateY(-1px);
#     }

#     [data-testid="stButton"] > button[kind="primary"] {
#         min-height: 2.85rem;
#         padding: 0.7rem 1rem;
#         border-color: #8998D2 !important;
#         background: #5B6CA8 !important;
#         color: #F8FAFC !important;
#         font-weight: 800;
#         box-shadow: 0 5px 14px rgba(91, 108, 168, 0.2);
#     }

#     [data-testid="stButton"] > button[kind="primary"]:hover {
#         border-color: #A8B5E8 !important;
#         background: #6E80BD !important;
#         box-shadow: 0 8px 18px rgba(91, 108, 168, 0.3);
#     }

#     [data-testid="stButton"] > button:focus-visible {
#         outline: none;
#         box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.5);
#     }

#     [data-testid="stExpander"] {
#         border: 1px solid var(--border) !important;
#         border-left: 3px solid var(--primary-accent) !important;
#         border-radius: 12px;
#         background: rgba(23, 35, 56, 0.72);
#         overflow: hidden;
#     }

#     [data-testid="stExpander"] .streamlit-expanderHeader {
#         background: rgba(17, 27, 46, 0.9);
#         color: var(--text);
#         border-bottom: 1px solid rgba(51, 65, 85, 0.7);
#         font-weight: 600;
#         padding: 0.9rem 1rem;
#     }

#     [data-testid="stExpander"] .streamlit-expanderHeader:hover {
#         color: var(--text);
#         background: rgba(23, 35, 56, 0.9);
#     }

#     .answer-label {
#         margin-bottom: 0.75rem;
#     }

#     .metadata-label {
#         display: block;
#         margin-bottom: 0.25rem;
#         color: var(--muted);
#         font-size: 0.68rem;
#         font-weight: 700;
#         letter-spacing: 0.08em;
#         text-transform: uppercase;
#     }

#     .workflow-track {
#         display: flex;
#         align-items: center;
#         gap: 0.75rem;
#         padding: 0.3rem 0;
#     }

#     .workflow-step {
#         display: flex;
#         align-items: center;
#         gap: 0.8rem;
#         flex: 1 1 0;
#         min-width: 0;
#         padding: 0.85rem 0.9rem;
#         border: 1px solid var(--border);
#         border-radius: 10px;
#         background: rgba(23, 35, 56, 0.7);
#     }

#     .workflow-number {
#         display: grid;
#         place-items: center;
#         flex: 0 0 30px;
#         width: 30px;
#         height: 30px;
#         border: 1px solid rgba(167, 139, 250, 0.45);
#         border-radius: 9px;
#         background: rgba(167, 139, 250, 0.08);
#         color: var(--primary-accent);
#         font-size: 0.72rem;
#         font-weight: 800;
#     }

#     .workflow-name {
#         display: block;
#         color: var(--text);
#         font-size: 0.88rem;
#         font-weight: 700;
#     }

#     .workflow-detail {
#         display: block;
#         margin-top: 0.14rem;
#         color: var(--muted);
#         font-size: 0.72rem;
#     }

#     .workflow-link {
#         flex: 0 0 28px;
#         height: 1px;
#         background: rgba(148, 163, 184, 0.4);
#     }

#     .benchmark-shell {
#         margin-top: 0.5rem;
#         border: 1px solid var(--border);
#         border-radius: 14px;
#         background: rgba(17, 27, 46, 0.96);
#         overflow: hidden;
#     }

#     .benchmark-table {
#         width: 100%;
#         border-collapse: collapse;
#         table-layout: fixed;
#         color: var(--text);
#         background: rgba(17, 27, 46, 0.96);
#     }

#     .benchmark-table th {
#         padding: 0.95rem 1rem;
#         text-align: left;
#         color: var(--text);
#         font-size: 0.75rem;
#         font-weight: 700;
#         letter-spacing: 0.06em;
#         text-transform: uppercase;
#         background: rgba(23, 35, 56, 0.96);
#         border-bottom: 1px solid var(--border);
#     }

#     .benchmark-table td {
#         padding: 0.95rem 1rem;
#         color: var(--text);
#         font-size: 0.95rem;
#         line-height: 1.5;
#         border-bottom: 1px solid rgba(51, 65, 85, 0.9);
#     }

#     .benchmark-table tbody tr:nth-child(even) {
#         background: rgba(148, 163, 184, 0.025);
#     }

#     .benchmark-table tbody tr:last-child td {
#         border-bottom: none;
#     }

#     .metric-cell {
#         font-variant-numeric: tabular-nums;
#         text-align: right;
#         color: var(--secondary);
#     }

#     .metric-best-purple {
#         background: rgba(167, 139, 250, 0.13);
#         color: var(--text);
#         font-weight: 700;
#     }

#     .metric-best-teal {
#         background: rgba(45, 212, 191, 0.12);
#         color: var(--text);
#         font-weight: 700;
#     }

#     .metric-low-latency {
#         background: rgba(56, 189, 248, 0.08);
#         color: var(--text);
#     }

#     [data-testid="stAlert"], .stAlert {
#         border-radius: 10px;
#     }

#     hr {
#         border-color: var(--border);
#     }

#     @media (max-width: 760px) {
#         [data-testid="stMainBlockContainer"] {
#             padding: 4.5rem 1rem 2.5rem;
#         }

#         .brand-bar {
#             align-items: flex-start;
#             flex-direction: column;
#         }

#         .hero {
#             padding: 1.6rem 0 1rem;
#         }

#         .workflow-track {
#             align-items: stretch;
#             flex-direction: column;
#         }

#         .workflow-link {
#             flex: 0 0 16px;
#             width: 1px;
#             height: 16px;
#             margin-left: 1.8rem;
#         }

#         .benchmark-table {
#             table-layout: auto;
#         }
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# st.markdown(
#     """
#     <header class="brand-bar">
#         <div class="brand-lockup">
#             <div class="brand-mark" aria-hidden="true">
#                 <svg width="25" height="25" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
#                     <path d="M4.5 6.5C8.2 5.4 11.4 6.2 14 8.4V22C11.4 19.8 8.2 19 4.5 20.1V6.5Z" stroke="#A78BFA" stroke-width="1.7" stroke-linejoin="round"/>
#                     <path d="M23.5 6.5C19.8 5.4 16.6 6.2 14 8.4V22C16.6 19.8 19.8 19 23.5 20.1V6.5Z" stroke="#38BDF8" stroke-width="1.7" stroke-linejoin="round"/>
#                     <path d="M8 10.2C9.5 10.1 10.7 10.5 11.7 11.1M20 10.2C18.5 10.1 17.3 10.5 16.3 11.1" stroke="#F8FAFC" stroke-width="1.2" stroke-linecap="round"/>
#                 </svg>
#             </div>
#             <div>
#                 <div class="brand-name">ResearchIQ</div>
#                 <div class="brand-subtitle">Evidence-driven AI research assistant</div>
#             </div>
#         </div>
#         <div class="workspace-badge"><span class="badge-dot"></span>Academic Research Workspace</div>
#     </header>
#     <section class="hero">
#         <div class="hero-kicker">Research workspace</div>
#         <h1>Research deeper. Cite with confidence.</h1>
#         <p>Ask a research question, explore retrieved academic evidence, and review the passages supporting the answer.</p>
#     </section>
#     """,
#     unsafe_allow_html=True,
# )

# st.markdown('<div class="section-kicker">01 / ASK</div>', unsafe_allow_html=True)
# st.markdown('<h2 class="section-title">Start with a research question</h2>', unsafe_allow_html=True)
# st.markdown(
#     '<p class="section-description">Choose a starting point or ask a focused question of your own.</p>',
#     unsafe_allow_html=True,
# )

# example_columns = st.columns(3)
# for index, (column, (topic, example)) in enumerate(zip(example_columns, EXAMPLE_QUESTIONS)):
#     with column:
#         with st.container(border=True):
#             st.markdown(f'<div class="topic-label">{topic}</div>', unsafe_allow_html=True)
#             st.markdown(f'<div class="example-question">{example}</div>', unsafe_allow_html=True)
#             if st.button("Use question", key=f"example_{index}", use_container_width=True):
#                 select_example(example)
#                 st.rerun()

# with st.container(border=True):
#     st.markdown('<div class="section-kicker">YOUR INQUIRY</div>', unsafe_allow_html=True)
#     question = st.text_area(
#         "Research question",
#         key="research_question",
#         height=130,
#         placeholder="Ask about a method, finding, comparison, or limitation...",
#     )
#     run_research = st.button("Run Research", type="primary", use_container_width=True)

# if run_research:
#     st.session_state.pop("research_result", None)
#     st.session_state.pop("research_error", None)
#     if not question.strip():
#         st.session_state["research_error"] = "Enter a research question to continue."
#     else:
#         try:
#             with st.spinner("Retrieving and synthesizing evidence..."):
#                 response = httpx.post(
#                     RESEARCH_ENDPOINT,
#                     json={"question": question.strip()},
#                     timeout=httpx.Timeout(
#                         API_RESPONSE_TIMEOUT_SECONDS,
#                         connect=API_CONNECT_TIMEOUT_SECONDS,
#                     ),
#                 )
#                 response.raise_for_status()
#                 result = response.json()
#                 if not isinstance(result, dict):
#                     raise ValueError("The response body was not a JSON object.")
#         except httpx.HTTPStatusError as error:
#             try:
#                 body = error.response.json()
#                 detail = body.get("detail") if isinstance(body, dict) else None
#             except ValueError:
#                 detail = None
#             detail = str(detail or error.response.text).strip()
#             st.session_state["research_error"] = (
#                 f"Research request failed (HTTP {error.response.status_code}): {detail}"
#             )
#         except httpx.ConnectError:
#             st.session_state["research_error"] = (
#                 "Could not connect to the ResearchIQ API. Confirm the FastAPI service is running, then retry."
#             )
#         except httpx.TimeoutException as error:
#             if isinstance(error, httpx.ConnectTimeout):
#                 message = "The connection to the ResearchIQ API timed out. Confirm the FastAPI service is reachable."
#             else:
#                 message = (
#                     "The API did not respond within 120 seconds. arXiv discovery, indexing, or Gemini generation "
#                     "may still be running; check the API service before retrying."
#                 )
#             st.session_state["research_error"] = message
#         except httpx.RequestError:
#             st.session_state["research_error"] = (
#                 "The request failed while communicating with the ResearchIQ API. Check the service and retry."
#             )
#         except ValueError:
#             st.session_state["research_error"] = "The ResearchIQ API returned an invalid response."
#         else:
#             st.session_state["research_result"] = result
#             st.session_state["research_result_question"] = question.strip()

# research_error = st.session_state.get("research_error")
# if research_error:
#     st.markdown('<div class="section-kicker">REQUEST STATUS</div>', unsafe_allow_html=True)
#     with st.container(border=True):
#         st.markdown("#### Research request could not be completed")
#         st.write(research_error)

# result = st.session_state.get("research_result")
# if result:
#     st.divider()
#     st.markdown('<div class="section-kicker">02 / FINDINGS</div>', unsafe_allow_html=True)
#     st.markdown('<h2 class="section-title">Research results</h2>', unsafe_allow_html=True)
#     st.caption(f"Question: {st.session_state.get('research_result_question', '')}")

#     with st.container(border=True):
#         st.markdown('<div class="answer-label">SYNTHESIS</div>', unsafe_allow_html=True)
#         st.markdown(result.get("answer", "No answer returned."))

#     sources = result.get("sources", [])
#     st.markdown('<h3 class="section-title">Evidence</h3>', unsafe_allow_html=True)
#     if not sources:
#         st.info("No sources were returned for this question.")
#     for source in sources:
#         citation = source.get("citation_id", "[?]")
#         raw_paper_id = source.get("paper_id")
#         paper_id = raw_paper_id.strip() if isinstance(raw_paper_id, str) else ""
#         arxiv_label = f"arXiv:{paper_id}" if paper_id else "arXiv ID unavailable"
#         raw_title = source.get("title")
#         title = raw_title.strip() if isinstance(raw_title, str) and raw_title.strip() else f"Title unavailable · {arxiv_label}"
#         with st.expander(f"{citation}  ·  {title}  ·  {arxiv_label}"):
#             authors = source.get("authors", [])
#             author_text = ", ".join(str(author) for author in authors) if isinstance(authors, list) else ""
#             if author_text:
#                 st.caption(author_text)
#             paper_url = source.get("paper_url")
#             if isinstance(paper_url, str) and paper_url.startswith("https://arxiv.org/abs/"):
#                 st.link_button("Open original paper on arXiv", paper_url)
#             section_column, page_column, method_column = st.columns([2, 1, 2])
#             with section_column:
#                 st.markdown('<span class="metadata-label">Section</span>', unsafe_allow_html=True)
#                 st.write(source.get("section", "Unknown"))
#             with page_column:
#                 st.markdown('<span class="metadata-label">Page</span>', unsafe_allow_html=True)
#                 st.write(source.get("page", "Unknown"))
#             with method_column:
#                 st.markdown('<span class="metadata-label">Retrieval method</span>', unsafe_allow_html=True)
#                 st.write(source.get("retrieval_method", "Unknown"))
#             st.markdown("**Relevant evidence**")
#             st.write(source.get("text", ""))

# st.divider()
# st.markdown('<div class="section-kicker">03 / METHOD</div>', unsafe_allow_html=True)
# st.markdown('<h2 class="section-title">Research process</h2>', unsafe_allow_html=True)
# st.markdown(
#     '<p class="section-description">Static overview; not live execution status.</p>',
#     unsafe_allow_html=True,
# )
# st.markdown(
#     """
#     <div class="workflow-track" aria-label="Planner, researcher with hybrid retrieval, then critic">
#         <div class="workflow-step"><span class="workflow-number">01</span><span><span class="workflow-name">Planner</span><span class="workflow-detail">Frame the question</span></span></div>
#         <span class="workflow-link" aria-hidden="true"></span>
#         <div class="workflow-step"><span class="workflow-number">02</span><span><span class="workflow-name">Researcher</span><span class="workflow-detail">Hybrid retrieval</span></span></div>
#         <span class="workflow-link" aria-hidden="true"></span>
#         <div class="workflow-step"><span class="workflow-number">03</span><span><span class="workflow-name">Critic</span><span class="workflow-detail">Review grounding</span></span></div>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# benchmark = load_retrieval_benchmark()
# st.divider()
# st.markdown(
#     '<p class="section-description">Curious how retrieval performs? Explore the benchmark and evaluation limits.</p>',
#     unsafe_allow_html=True,
# )
# with st.expander("View retrieval benchmark & methodology", expanded=False):
#     st.markdown('<div class="section-kicker">04 / EVALUATION</div>', unsafe_allow_html=True)
#     st.markdown('<h2 class="section-title">Retrieval benchmark</h2>', unsafe_allow_html=True)
#     if benchmark:
#         st.markdown(
#             f'<p class="section-description">This compares how retrieval strategies find relevant research evidence. '
#             f'Recall@5 is the share of annotated relevant passages found in the first five results; '
#             f'MRR@5 rewards placing a relevant passage near the top. K=5 · {benchmark["question_count"]} manually '
#             "annotated questions · one paper: <em>Attention Is All You Need</em>.</p>"
#             '<p class="section-description"><strong>Trade-off:</strong> the cross-encoder reranker scored highest '
#             'on this evaluation, while adding latency (1.0722 s versus 0.0070 s for Hybrid RRF). These limited '
#             'results are not a general accuracy guarantee and do not measure factual accuracy or hallucination reduction.</p>',
#             unsafe_allow_html=True,
#         )
#         render_benchmark_table(benchmark)
#     else:
#         st.warning("Retrieval benchmark data is missing or invalid; no benchmark values are shown.")

# st.divider()
# st.markdown('<h2 class="section-title">About ResearchIQ</h2>', unsafe_allow_html=True)
# st.markdown(
#     """
#     - Structure-aware paper ingestion turns source PDFs into section-aware chunks.
#     - Dense retrieval and BM25 provide semantic and lexical candidates.
#     - Reciprocal Rank Fusion combines the candidate rankings.
#     - A cross-encoder reranks the fused evidence.
#     - Gemini provides answer synthesis by default.
#     - Returned source passages and citation labels keep the evidence reviewable.
#     """
# )
# st.markdown('<footer class="site-footer">Built by Siddhi · ResearchIQ</footer>', unsafe_allow_html=True)
"""Streamlit research interface for ResearchIQ."""

"""Streamlit research interface for ResearchIQ."""

import json
import logging
import math
import os
import sys
from html import escape
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EVALUATION_RESULTS_PATH = PROJECT_ROOT / "data/eval/results/retrieval_evaluation_results.json"
EVALUATION_DATASET_PATH = PROJECT_ROOT / "data/eval/attention_transformer_eval.json"
LOGGER = logging.getLogger(__name__)


def run_local_research(question: str) -> dict[str, object]:
    """Run retrieval and generation inside Streamlit without calling Render."""
    # Streamlit secrets must be put into the environment before importing app.config,
    # because app.config reads its settings when that module is first imported.
    for secret_name in (
        "GEMINI_API_KEY",
        "LLM_API_KEY",
        "LLM_PROVIDER",
        "GEMINI_MODEL",
        "LLM_MODEL",
        "LLM_API_BASE_URL",
    ):
        if os.getenv(secret_name):
            continue
        try:
            secret_value = st.secrets.get(secret_name)
        except Exception:
            secret_value = None
        if secret_value:
            os.environ[secret_name] = str(secret_value)

    # Delayed import is intentional: load configuration only after secrets are set.
    from app.api.research import get_rag_pipeline
    from app.ingestion.paper_service import normalize_arxiv_id

    pipeline = get_rag_pipeline()
    answer = pipeline.answer_question(question, top_k=5)

    sources: list[dict[str, object]] = []
    for source in answer.sources:
        paper_id = str(getattr(source, "paper_id", "") or "")
        paper_url = None
        if paper_id:
            try:
                normalized_id = normalize_arxiv_id(paper_id)
                paper_url = f"https://arxiv.org/abs/{normalized_id}"
            except ValueError:
                paper_url = None

        sources.append(
            {
                "citation_id": f"[{source.citation_id}]",
                "chunk_id": source.chunk_id,
                "paper_id": paper_id,
                "paper_url": paper_url,
                "title": source.title,
                "authors": source.authors,
                "section": source.section,
                "page": source.page,
                "text": source.text,
                "similarity_score": source.similarity_score,
                "retrieval_score": source.retrieval_score,
                "retrieval_method": source.retrieval_method,
            }
        )

    return {"answer": answer.answer, "sources": sources}

EXAMPLE_QUESTIONS = (
    (
        "Retrieval & grounding",
        "How does retrieval-augmented generation reduce hallucinations in large language models?",
    ),
    (
        "Deep reinforcement learning",
        "Which techniques improve sample efficiency in deep reinforcement learning?",
    ),
    (
        "Vision architectures",
        "How do vision transformers compare with convolutional networks for image recognition?",
    ),
)

EVALUATION_METHODS = (
    ("dense", "Dense retrieval"),
    ("bm25", "BM25"),
    ("hybrid_rrf", "Hybrid RRF"),
    ("hybrid_rrf_reranker", "Hybrid RRF + Cross-Encoder Reranker"),
)


def select_example(question: str) -> None:
    st.session_state["research_question"] = question


def load_retrieval_benchmark() -> dict[str, object] | None:
    try:
        results = json.loads(EVALUATION_RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(results, list):
        return None

    try:
        dataset = json.loads(EVALUATION_DATASET_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(dataset, list) or len(dataset) != 12:
        return None
    if any(
        not isinstance(question, dict)
        or not isinstance(question.get("question"), str)
        or not question["question"].strip()
        or not isinstance(question.get("relevant_chunk_ids"), list)
        or not question["relevant_chunk_ids"]
        or any(
            not isinstance(chunk_id, str) or "-chunk-" not in chunk_id
            for chunk_id in question["relevant_chunk_ids"]
        )
        for question in dataset
    ):
        return None
    paper_ids = {
        chunk_id.rsplit("-chunk-", 1)[0]
        for question in dataset
        for chunk_id in question["relevant_chunk_ids"]
        if isinstance(chunk_id, str) and "-chunk-" in chunk_id
    }
    if len(paper_ids) != 1:
        return None

    method_labels = dict(EVALUATION_METHODS)
    results_by_method = {}
    for result in results:
        if not isinstance(result, dict) or result.get("K") != 5:
            continue
        method = result.get("method")
        if not isinstance(method, str):
            return None
        if method not in method_labels:
            continue
        if method in results_by_method:
            return None
        try:
            recall = result["Recall@K"]
            mrr = result["MRR@K"]
            latency = result["average_latency"]
        except (KeyError, TypeError, ValueError):
            return None
        raw_metrics = (recall, mrr, latency)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_metrics):
            return None
        try:
            recall, mrr, latency = (float(value) for value in raw_metrics)
        except (OverflowError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (recall, mrr, latency)):
            return None
        if not 0 <= recall <= 1 or not 0 <= mrr <= 1 or latency < 0:
            return None
        results_by_method[method] = {
            "Retrieval method": method_labels[method],
            "Recall@5": recall,
            "MRR@5": mrr,
            "Average latency (s)": latency,
        }

    if set(results_by_method) != set(method_labels):
        return None
    return {
        "rows": [results_by_method[method] for method, _ in EVALUATION_METHODS],
        "question_count": len(dataset),
    }


def format_latency(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds:.6f}" if seconds >= 0.000001 else f"{seconds:.2e}"
    return f"{seconds:.4f}"


def render_benchmark_table(benchmark: dict[str, object]) -> None:
    rows = benchmark.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return

    numeric_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        numeric_rows.append(
            {
                "method": str(row.get("Retrieval method", "Unknown")),
                "recall": float(row.get("Recall@5", 0.0)),
                "mrr": float(row.get("MRR@5", 0.0)),
                "latency": float(row.get("Average latency (s)", 0.0)),
            }
        )

    if not numeric_rows:
        return

    best_recall = max(item["recall"] for item in numeric_rows)
    best_mrr = max(item["mrr"] for item in numeric_rows)
    lowest_latency = min(item["latency"] for item in numeric_rows)

    table_rows = []
    for item in numeric_rows:
        recall_class = " metric-best-purple" if abs(item["recall"] - best_recall) < 1e-12 else ""
        mrr_class = " metric-best-teal" if abs(item["mrr"] - best_mrr) < 1e-12 else ""
        latency_class = " metric-low-latency" if abs(item["latency"] - lowest_latency) < 1e-12 else ""

        method = escape(item["method"])
        recall = f"{item['recall']:.3f}"
        mrr = f"{item['mrr']:.3f}"
        latency = f"{format_latency(item['latency'])} s"

        table_rows.append(
            "<tr>"
            f"<td>{method}</td>"
            f"<td class=\"metric-cell{recall_class}\">{recall}</td>"
            f"<td class=\"metric-cell{mrr_class}\">{mrr}</td>"
            f"<td class=\"metric-cell{latency_class}\">{latency}</td>"
            "</tr>"
        )

    table_html = "".join(table_rows)
    st.markdown(
        f"""
        <div class="benchmark-shell">
            <table class="benchmark-table">
                <thead>
                    <tr>
                        <th>Retrieval method</th>
                        <th>Recall@5</th>
                        <th>MRR@5</th>
                        <th>Average latency</th>
                    </tr>
                </thead>
                <tbody>
                    {table_html}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="ResearchIQ | Research Workspace",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --page: #0B1120;
        --panel: #111B2E;
        --panel-raised: #172338;
        --surface: #0F172A;
        --text: #F8FAFC;
        --secondary: #CBD5E1;
        --muted: #94A3B8;
        --border: #334155;
        --primary-accent: #A78BFA;
        --secondary-accent: #38BDF8;
        --success-accent: #2DD4BF;
        --shadow: rgba(2, 6, 23, 0.52);
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--page);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, 0.88);
        border-bottom: 1px solid rgba(51, 65, 85, 0.7);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1200px;
        padding: 4.5rem 2.25rem 4rem;
    }

    h1, h2, h3, h4, p, label, li, div, span { color: var(--text); }

    h1, h2, h3, h4 {
        letter-spacing: -0.02em;
        color: var(--text);
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"],
    .stCaption,
    .section-description,
    .metadata-label,
    [data-testid="stHorizontalBlock"] span,
    [data-testid="stExpander"] .streamlit-expanderHeader {
        color: var(--secondary);
    }

    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding-bottom: 1.25rem;
        border-bottom: 1px solid var(--border);
    }

    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }

    .brand-mark {
        display: grid;
        place-items: center;
        width: 44px;
        height: 44px;
        border: 1px solid rgba(167, 139, 250, 0.42);
        border-radius: 12px;
        background: linear-gradient(145deg, rgba(167, 139, 250, 0.12), rgba(56, 189, 248, 0.08));
        box-shadow: 0 8px 20px rgba(167, 139, 250, 0.12);
    }

    .brand-name {
        color: var(--text);
        font-size: 1.55rem;
        line-height: 1.1;
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .brand-subtitle {
        margin-top: 0.14rem;
        color: var(--secondary);
        font-size: 0.78rem;
        line-height: 1.3;
    }

    .workspace-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.45rem 0.75rem;
        border: 1px solid rgba(167, 139, 250, 0.24);
        border-radius: 999px;
        background: rgba(167, 139, 250, 0.06);
        color: var(--secondary);
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .badge-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--primary-accent);
        box-shadow: 0 0 10px rgba(167, 139, 250, 0.7);
    }

    .hero {
        max-width: 780px;
        padding: 2rem 0 1.4rem;
    }

    .hero-kicker,
    .section-kicker,
    .topic-label,
    .answer-label {
        color: var(--primary-accent);
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }

    .hero h1 {
        margin: 0.45rem 0 0.7rem;
        color: var(--text);
        font-size: clamp(2.5rem, 3.8vw, 3.5rem);
        line-height: 1.04;
        font-weight: 800;
        letter-spacing: -0.05em;
    }

    .hero p {
        max-width: 62ch;
        margin: 0;
        color: var(--secondary);
        font-size: 1rem;
        line-height: 1.7;
    }

    .section-title {
        margin: 0.2rem 0 0.35rem;
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text);
    }

    .section-description {
        margin: 0 0 0.9rem;
        color: var(--secondary);
        font-size: 0.94rem;
        line-height: 1.6;
    }

    .site-footer {
        margin-top: 1rem;
        padding: 0.75rem 0;
        color: var(--muted);
        font-size: 0.78rem;
        text-align: center;
    }

    [data-testid="stVerticalBlockBorderWrapper"] > div {
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        background: linear-gradient(180deg, rgba(17, 27, 46, 0.96), rgba(17, 27, 46, 0.9)) !important;
        box-shadow: none;
        transition: border-color 150ms ease, box-shadow 150ms ease, transform 150ms ease;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:hover > div {
        border-color: rgba(167, 139, 250, 0.68) !important;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.36);
    }

    .topic-label {
        margin-bottom: 0.5rem;
        color: var(--secondary-accent);
        font-size: 0.68rem;
    }

    .example-question {
        min-height: 4.5rem;
        color: var(--text);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    [data-testid="stTextArea"] textarea {
        min-height: 136px;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: rgba(11, 17, 32, 0.7);
        color: var(--text);
        line-height: 1.6;
        resize: vertical;
    }

    [data-testid="stTextArea"] textarea:focus {
        border-color: rgba(167, 139, 250, 0.9);
        box-shadow: 0 0 0 1px rgba(167, 139, 250, 0.28);
    }

    [data-testid="stTextArea"] textarea::placeholder {
        color: var(--muted);
    }

    [data-testid="stButton"] > button {
        min-height: 2.7rem;
        border: 1px solid rgba(167, 139, 250, 0.25);
        border-radius: 10px;
        background: rgba(23, 35, 56, 0.9);
        color: var(--text);
        font-weight: 700;
        transition: background 150ms ease, border-color 150ms ease, transform 150ms ease, box-shadow 150ms ease;
    }

    [data-testid="stButton"] > button:hover {
        border-color: rgba(167, 139, 250, 0.55);
        background: rgba(34, 46, 67, 0.96);
        color: var(--text);
        transform: translateY(-1px);
    }

    [data-testid="stButton"] > button[kind="primary"] {
        min-height: 2.85rem;
        padding: 0.7rem 1rem;
        border-color: #8998D2 !important;
        background: #5B6CA8 !important;
        color: #F8FAFC !important;
        font-weight: 800;
        box-shadow: 0 5px 14px rgba(91, 108, 168, 0.2);
    }

    [data-testid="stButton"] > button[kind="primary"]:hover {
        border-color: #A8B5E8 !important;
        background: #6E80BD !important;
        box-shadow: 0 8px 18px rgba(91, 108, 168, 0.3);
    }

    [data-testid="stButton"] > button:focus-visible {
        outline: none;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.5);
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-left: 3px solid var(--primary-accent) !important;
        border-radius: 12px;
        background: rgba(23, 35, 56, 0.72);
        overflow: hidden;
    }

    [data-testid="stExpander"] .streamlit-expanderHeader {
        background: rgba(17, 27, 46, 0.9);
        color: var(--text);
        border-bottom: 1px solid rgba(51, 65, 85, 0.7);
        font-weight: 600;
        padding: 0.9rem 1rem;
    }

    [data-testid="stExpander"] .streamlit-expanderHeader:hover {
        color: var(--text);
        background: rgba(23, 35, 56, 0.9);
    }

    .answer-label {
        margin-bottom: 0.75rem;
    }

    .metadata-label {
        display: block;
        margin-bottom: 0.25rem;
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .workflow-track {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.3rem 0;
    }

    .workflow-step {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        flex: 1 1 0;
        min-width: 0;
        padding: 0.85rem 0.9rem;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: rgba(23, 35, 56, 0.7);
    }

    .workflow-number {
        display: grid;
        place-items: center;
        flex: 0 0 30px;
        width: 30px;
        height: 30px;
        border: 1px solid rgba(167, 139, 250, 0.45);
        border-radius: 9px;
        background: rgba(167, 139, 250, 0.08);
        color: var(--primary-accent);
        font-size: 0.72rem;
        font-weight: 800;
    }

    .workflow-name {
        display: block;
        color: var(--text);
        font-size: 0.88rem;
        font-weight: 700;
    }

    .workflow-detail {
        display: block;
        margin-top: 0.14rem;
        color: var(--muted);
        font-size: 0.72rem;
    }

    .workflow-link {
        flex: 0 0 28px;
        height: 1px;
        background: rgba(148, 163, 184, 0.4);
    }

    .benchmark-shell {
        margin-top: 0.5rem;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(17, 27, 46, 0.96);
        overflow: hidden;
    }

    .benchmark-table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        color: var(--text);
        background: rgba(17, 27, 46, 0.96);
    }

    .benchmark-table th {
        padding: 0.95rem 1rem;
        text-align: left;
        color: var(--text);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        background: rgba(23, 35, 56, 0.96);
        border-bottom: 1px solid var(--border);
    }

    .benchmark-table td {
        padding: 0.95rem 1rem;
        color: var(--text);
        font-size: 0.95rem;
        line-height: 1.5;
        border-bottom: 1px solid rgba(51, 65, 85, 0.9);
    }

    .benchmark-table tbody tr:nth-child(even) {
        background: rgba(148, 163, 184, 0.025);
    }

    .benchmark-table tbody tr:last-child td {
        border-bottom: none;
    }

    .metric-cell {
        font-variant-numeric: tabular-nums;
        text-align: right;
        color: var(--secondary);
    }

    .metric-best-purple {
        background: rgba(167, 139, 250, 0.13);
        color: var(--text);
        font-weight: 700;
    }

    .metric-best-teal {
        background: rgba(45, 212, 191, 0.12);
        color: var(--text);
        font-weight: 700;
    }

    .metric-low-latency {
        background: rgba(56, 189, 248, 0.08);
        color: var(--text);
    }

    [data-testid="stAlert"], .stAlert {
        border-radius: 10px;
    }

    hr {
        border-color: var(--border);
    }

    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] {
            padding: 4.5rem 1rem 2.5rem;
        }

        .brand-bar {
            align-items: flex-start;
            flex-direction: column;
        }

        .hero {
            padding: 1.6rem 0 1rem;
        }

        .workflow-track {
            align-items: stretch;
            flex-direction: column;
        }

        .workflow-link {
            flex: 0 0 16px;
            width: 1px;
            height: 16px;
            margin-left: 1.8rem;
        }

        .benchmark-table {
            table-layout: auto;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <header class="brand-bar">
        <div class="brand-lockup">
            <div class="brand-mark" aria-hidden="true">
                <svg width="25" height="25" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M4.5 6.5C8.2 5.4 11.4 6.2 14 8.4V22C11.4 19.8 8.2 19 4.5 20.1V6.5Z" stroke="#A78BFA" stroke-width="1.7" stroke-linejoin="round"/>
                    <path d="M23.5 6.5C19.8 5.4 16.6 6.2 14 8.4V22C16.6 19.8 19.8 19 23.5 20.1V6.5Z" stroke="#38BDF8" stroke-width="1.7" stroke-linejoin="round"/>
                    <path d="M8 10.2C9.5 10.1 10.7 10.5 11.7 11.1M20 10.2C18.5 10.1 17.3 10.5 16.3 11.1" stroke="#F8FAFC" stroke-width="1.2" stroke-linecap="round"/>
                </svg>
            </div>
            <div>
                <div class="brand-name">ResearchIQ</div>
                <div class="brand-subtitle">Evidence-driven AI research assistant</div>
            </div>
        </div>
        <div class="workspace-badge"><span class="badge-dot"></span>Academic Research Workspace</div>
    </header>
    <section class="hero">
        <div class="hero-kicker">Research workspace</div>
        <h1>Research deeper. Cite with confidence.</h1>
        <p>Ask a research question, explore retrieved academic evidence, and review the passages supporting the answer.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">01 / ASK</div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Start with a research question</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-description">Choose a starting point or ask a focused question of your own.</p>',
    unsafe_allow_html=True,
)

example_columns = st.columns(3)
for index, (column, (topic, example)) in enumerate(zip(example_columns, EXAMPLE_QUESTIONS)):
    with column:
        with st.container(border=True):
            st.markdown(f'<div class="topic-label">{topic}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="example-question">{example}</div>', unsafe_allow_html=True)
            if st.button("Use question", key=f"example_{index}", use_container_width=True):
                select_example(example)
                st.rerun()

with st.container(border=True):
    st.markdown('<div class="section-kicker">YOUR INQUIRY</div>', unsafe_allow_html=True)
    question = st.text_area(
        "Research question",
        key="research_question",
        height=130,
        placeholder="Ask about a method, finding, comparison, or limitation...",
    )
    run_research = st.button("Run Research", type="primary", use_container_width=True)

if run_research:
    st.session_state.pop("research_result", None)
    st.session_state.pop("research_error", None)
    if not question.strip():
        st.session_state["research_error"] = "Enter a research question to continue."
    else:
        try:
            with st.spinner("Retrieving and synthesizing evidence..."):
                result = run_local_research(question.strip())
        except Exception as error:
            LOGGER.exception("ResearchIQ local research request failed.")
            detail = getattr(error, "detail", None)
            if not isinstance(detail, str) or not detail.strip():
                detail = str(error).strip() or type(error).__name__
            st.session_state["research_error"] = f"Research request failed: {detail}"
        else:
            st.session_state["research_result"] = result
            st.session_state["research_result_question"] = question.strip()

research_error = st.session_state.get("research_error")
if research_error:
    st.markdown('<div class="section-kicker">REQUEST STATUS</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("#### Research request could not be completed")
        st.write(research_error)

result = st.session_state.get("research_result")
if result:
    st.divider()
    st.markdown('<div class="section-kicker">02 / FINDINGS</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Research results</h2>', unsafe_allow_html=True)
    st.caption(f"Question: {st.session_state.get('research_result_question', '')}")

    with st.container(border=True):
        st.markdown('<div class="answer-label">SYNTHESIS</div>', unsafe_allow_html=True)
        st.markdown(result.get("answer", "No answer returned."))

    sources = result.get("sources", [])
    st.markdown('<h3 class="section-title">Evidence</h3>', unsafe_allow_html=True)
    if not sources:
        st.info("No sources were returned for this question.")
    for source in sources:
        citation = source.get("citation_id", "[?]")
        raw_paper_id = source.get("paper_id")
        paper_id = raw_paper_id.strip() if isinstance(raw_paper_id, str) else ""
        arxiv_label = f"arXiv:{paper_id}" if paper_id else "arXiv ID unavailable"
        raw_title = source.get("title")
        title = raw_title.strip() if isinstance(raw_title, str) and raw_title.strip() else f"Title unavailable · {arxiv_label}"
        with st.expander(f"{citation}  ·  {title}  ·  {arxiv_label}"):
            authors = source.get("authors", [])
            author_text = ", ".join(str(author) for author in authors) if isinstance(authors, list) else ""
            if author_text:
                st.caption(author_text)
            paper_url = source.get("paper_url")
            if isinstance(paper_url, str) and paper_url.startswith("https://arxiv.org/abs/"):
                st.link_button("Open original paper on arXiv", paper_url)
            section_column, page_column, method_column = st.columns([2, 1, 2])
            with section_column:
                st.markdown('<span class="metadata-label">Section</span>', unsafe_allow_html=True)
                st.write(source.get("section", "Unknown"))
            with page_column:
                st.markdown('<span class="metadata-label">Page</span>', unsafe_allow_html=True)
                st.write(source.get("page", "Unknown"))
            with method_column:
                st.markdown('<span class="metadata-label">Retrieval method</span>', unsafe_allow_html=True)
                st.write(source.get("retrieval_method", "Unknown"))
            st.markdown("**Relevant evidence**")
            st.write(source.get("text", ""))

st.divider()
st.markdown('<div class="section-kicker">03 / METHOD</div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">Research process</h2>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-description">Static overview; not live execution status.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="workflow-track" aria-label="Planner, researcher with hybrid retrieval, then critic">
        <div class="workflow-step"><span class="workflow-number">01</span><span><span class="workflow-name">Planner</span><span class="workflow-detail">Frame the question</span></span></div>
        <span class="workflow-link" aria-hidden="true"></span>
        <div class="workflow-step"><span class="workflow-number">02</span><span><span class="workflow-name">Researcher</span><span class="workflow-detail">Hybrid retrieval</span></span></div>
        <span class="workflow-link" aria-hidden="true"></span>
        <div class="workflow-step"><span class="workflow-number">03</span><span><span class="workflow-name">Critic</span><span class="workflow-detail">Review grounding</span></span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

benchmark = load_retrieval_benchmark()
st.divider()
st.markdown(
    '<p class="section-description">Curious how retrieval performs? Explore the benchmark and evaluation limits.</p>',
    unsafe_allow_html=True,
)
with st.expander("View retrieval benchmark & methodology", expanded=False):
    st.markdown('<div class="section-kicker">04 / EVALUATION</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Retrieval benchmark</h2>', unsafe_allow_html=True)
    if benchmark:
        st.markdown(
            f'<p class="section-description">This compares how retrieval strategies find relevant research evidence. '
            f'Recall@5 is the share of annotated relevant passages found in the first five results; '
            f'MRR@5 rewards placing a relevant passage near the top. K=5 · {benchmark["question_count"]} manually '
            "annotated questions · one paper: <em>Attention Is All You Need</em>.</p>"
            '<p class="section-description"><strong>Trade-off:</strong> the cross-encoder reranker scored highest '
            'on this evaluation, while adding latency (1.0722 s versus 0.0070 s for Hybrid RRF). These limited '
            'results are not a general accuracy guarantee and do not measure factual accuracy or hallucination reduction.</p>',
            unsafe_allow_html=True,
        )
        render_benchmark_table(benchmark)
    else:
        st.warning("Retrieval benchmark data is missing or invalid; no benchmark values are shown.")

st.divider()
st.markdown('<h2 class="section-title">About ResearchIQ</h2>', unsafe_allow_html=True)
st.markdown(
    """
    - Structure-aware paper ingestion turns source PDFs into section-aware chunks.
    - Dense retrieval and BM25 provide semantic and lexical candidates.
    - Reciprocal Rank Fusion combines the candidate rankings.
    - The lightweight deployed configuration uses hybrid RRF retrieval without the cross-encoder reranker.
    - Gemini provides answer synthesis by default.
    - Returned source passages and citation labels keep the evidence reviewable.
    """
)
st.markdown('<footer class="site-footer">Built by Siddhi · ResearchIQ</footer>', unsafe_allow_html=True)