
# """Research query endpoint backed by the local vector index and LLM service."""

# import logging
# import re
# from contextlib import contextmanager
# from functools import lru_cache
# from pathlib import Path
# from time import perf_counter
# from typing import Iterator

# import httpx
# from fastapi import APIRouter, Depends, HTTPException

# from app.api.limits import enforce_research_rate_limit
# from app.api.schemas import (
#     ResearchQueryRequest,
#     ResearchQueryResponse,
#     ResearchSourceResponse,
# )
# from app.config import settings
# from app.ingestion.errors import MetadataFetchError, PDFDownloadError
# from app.ingestion.paper_service import normalize_arxiv_id, search_arxiv_papers
# from app.ingestion.pipeline import ingest_paper
# from app.ingestion.storage import DEFAULT_PAPERS_DIR, processed_path_for
# from app.retrieval.bm25_retriever import BM25Retriever
# from app.retrieval.embeddings import SentenceTransformerEmbedder
# from app.retrieval.hybrid_retriever import HybridRetriever
# from app.retrieval.index_builder import append_processed_chunks, build_index
# from app.retrieval.reranker import CrossEncoderReranker
# from app.retrieval.retriever import VectorRetriever
# from app.retrieval.vector_store import DEFAULT_INDEX_PATH, FaissVectorStore
# from app.services.llm_service import (
#     LLMConfigurationError,
#     LLMProviderUnavailableError,
#     LLMQuotaError,
#     LLMRequestError,
#     LLMTimeoutError,
#     create_llm_service,
# )
# from app.services.rag_pipeline import RAGPipeline

# logger = logging.getLogger(__name__)
# router = APIRouter(prefix="/api/v1/research", tags=["research"])


# @contextmanager
# def _log_stage(stage: str, **fields: object) -> Iterator[None]:
#     started = perf_counter()
#     details = " ".join(f"{name}={value}" for name, value in fields.items())

#     logger.info("research_stage_started stage=%s %s", stage, details)

#     try:
#         yield
#     except Exception as error:
#         logger.warning(
#             "research_stage_failed stage=%s elapsed_seconds=%.3f error_type=%s",
#             stage,
#             perf_counter() - started,
#             type(error).__name__,
#         )
#         raise
#     else:
#         logger.info(
#             "research_stage_completed stage=%s elapsed_seconds=%.3f %s",
#             stage,
#             perf_counter() - started,
#             details,
#         )


# def _question_to_arxiv_query(question: str) -> str:
#     cleaned = re.sub(r"[^a-zA-Z0-9\s-]", " ", question).strip()
#     words = [token for token in cleaned.lower().split() if len(token) > 2]
#     return " ".join(words[:8])


# def refresh_relevant_papers(
#     question: str,
#     papers_dir: Path = DEFAULT_PAPERS_DIR,
#     index_path: Path = DEFAULT_INDEX_PATH,
#     max_results: int | None = None,
# ) -> int:
#     """Discover relevant arXiv papers, ingest missing papers, and update the index."""
#     if max_results is None:
#         max_results = settings.arxiv_search_limit

#     limit = max(1, int(max_results))
#     query = _question_to_arxiv_query(question)

#     if not query:
#         return 0

#     try:
#         with _log_stage("arxiv_search", result_limit=limit):
#             candidates = search_arxiv_papers(query, max_results=limit)
#     except MetadataFetchError:
#         logger.warning(
#             "Could not discover arXiv papers for this research request."
#         )
#         return 0

#     if not candidates:
#         logger.info("No arXiv papers were returned for a research request.")
#         return 0

#     papers_dir = Path(papers_dir)
#     papers_dir.mkdir(parents=True, exist_ok=True)
#     ingested_count = 0

#     for paper in candidates:
#         processed_path = processed_path_for(paper.arxiv_id, papers_dir)

#         if processed_path.exists():
#             continue

#         try:
#             with _log_stage("paper_ingestion", paper_id=paper.arxiv_id):
#                 result = ingest_paper(
#                     paper.arxiv_id,
#                     data_dir=papers_dir,
#                     metadata=paper,
#                 )
#         except Exception as error:
#             if isinstance(error, PDFDownloadError) and isinstance(
#                 error.__cause__, (httpx.TimeoutException, TimeoutError)
#             ):
#                 raise HTTPException(
#                     status_code=504,
#                     detail=(
#                         "Downloading a discovered arXiv paper timed out. "
#                         "The existing index was left unchanged; retry later."
#                     ),
#                 ) from error

#             logger.warning(
#                 "Skipping arXiv paper %s due to ingestion failure "
#                 "(error_type=%s).",
#                 paper.arxiv_id,
#                 type(error).__name__,
#             )
#             continue

#         if not result.chunks:
#             logger.warning(
#                 "Skipping arXiv paper %s because extraction yielded no chunks.",
#                 paper.arxiv_id,
#             )
#             continue

#         ingested_count += len(result.chunks)

#     if ingested_count == 0:
#         if not index_path.exists() and any(papers_dir.glob("*.json")):
#             with _log_stage("initial_index_build"):
#                 build_index(papers_dir, index_path)

#             get_rag_pipeline.cache_clear()

#         return 0

#     try:
#         with _log_stage(
#             "incremental_index_update",
#             new_chunks=ingested_count,
#         ):
#             append_processed_chunks(papers_dir, index_path)
#     except (FileNotFoundError, ValueError, OSError) as error:
#         logger.warning(
#             "Updating the FAISS index failed after arXiv discovery: %s",
#             error,
#         )
#         return 0

#     get_rag_pipeline.cache_clear()
#     return ingested_count


# @lru_cache(maxsize=1)
# def get_rag_pipeline() -> RAGPipeline:
#     """Load the local retrieval index and construct the RAG pipeline."""
#     if not DEFAULT_INDEX_PATH.is_file():
#         processed_jsons = list(Path(DEFAULT_PAPERS_DIR).glob("*.json"))

#         if processed_jsons:
#             try:
#                 with _log_stage("initial_index_build"):
#                     build_index(DEFAULT_PAPERS_DIR, DEFAULT_INDEX_PATH)
#             except (FileNotFoundError, ValueError, OSError) as error:
#                 raise HTTPException(
#                     status_code=503,
#                     detail=(
#                         "The local retrieval index could not be prepared. "
#                         "Contact the demo maintainer."
#                     ),
#                 ) from error

#     try:
#         vector_store = FaissVectorStore.load(DEFAULT_INDEX_PATH)
#     except (FileNotFoundError, ValueError, OSError) as error:
#         raise HTTPException(
#             status_code=503,
#             detail=(
#                 "The local vector index is unavailable. "
#                 "No usable indexed papers are available yet."
#             ),
#         ) from error

#     dense_retriever = VectorRetriever(
#         vector_store,
#         SentenceTransformerEmbedder(),
#     )

#     hybrid_retriever = HybridRetriever(
#         dense_retriever=dense_retriever,
#         bm25_retriever=BM25Retriever(vector_store.chunks),
#         reranker=None,
#         dense_top_k=settings.dense_top_k,
#         bm25_top_k=settings.bm25_top_k,
#         rrf_k=settings.rrf_k,
#         rerank_top_k=settings.rerank_top_k,
#         final_top_k=settings.final_top_k,
#     )

#     return RAGPipeline(
#         retriever=hybrid_retriever,
#         llm=create_llm_service(),
#     )


# @router.post(
#     "/query",
#     response_model=ResearchQueryResponse,
#     dependencies=[Depends(enforce_research_rate_limit)],
# )
# def research_query(
#     request: ResearchQueryRequest,
# ) -> ResearchQueryResponse:
#     # Discover and index papers BEFORE loading the pipeline.
#     # This allows a fresh deployment to build its initial index.
#     with _log_stage(
#         "paper_discovery_and_indexing",
#         result_limit=settings.arxiv_search_limit,
#     ):
#         refresh_relevant_papers(
#             request.question,
#             max_results=settings.arxiv_search_limit,
#         )

#     # Load the pipeline only after the discovery/indexing step.
#     with _log_stage("retrieval_pipeline_load"):
#         pipeline = get_rag_pipeline()

#     try:
#         with _log_stage(
#             "retrieval_and_generation",
#             top_k=request.top_k,
#         ):
#             result = pipeline.answer_question(
#                 request.question,
#                 top_k=request.top_k,
#             )

#     except LLMConfigurationError as error:
#         raise HTTPException(
#             status_code=503,
#             detail=str(error),
#         ) from error

#     except LLMProviderUnavailableError as error:
#         raise HTTPException(
#             status_code=503,
#             detail=str(error),
#         ) from error

#     except LLMTimeoutError as error:
#         raise HTTPException(
#             status_code=504,
#             detail=str(error),
#         ) from error

#     except LLMQuotaError as error:
#         raise HTTPException(
#             status_code=429,
#             detail=str(error),
#         ) from error

#     except LLMRequestError as error:
#         raise HTTPException(
#             status_code=502,
#             detail=str(error),
#         ) from error

#     if not result.sources:
#         result = type(result)(
#             answer=(
#                 "I could not find relevant academic evidence for your question "
#                 "in the indexed papers. The arXiv discovery step did not yield "
#                 "usable papers or the current index has no matching chunks."
#             ),
#             sources=[],
#         )

#     return ResearchQueryResponse(
#         answer=result.answer,
#         sources=[
#             ResearchSourceResponse(
#                 citation_id=f"[{source.citation_id}]",
#                 chunk_id=source.chunk_id,
#                 paper_id=source.paper_id,
#                 paper_url=_paper_url(source.paper_id),
#                 title=source.title,
#                 authors=source.authors,
#                 section=source.section,
#                 page=source.page,
#                 text=source.text,
#                 similarity_score=source.similarity_score,
#                 retrieval_score=source.retrieval_score,
#                 retrieval_method=source.retrieval_method,
#             )
#             for source in result.sources
#         ],
#     )


# def _paper_url(paper_id: str) -> str | None:
#     try:
#         normalized_id = normalize_arxiv_id(paper_id)
#     except ValueError:    
#         return None

#     return f"https://arxiv.org/abs/{normalized_id}"

"""Research query endpoint using a prebuilt FAISS index and LLM service."""

import logging
from contextlib import contextmanager
from functools import lru_cache
from time import perf_counter
from typing import Iterator

from fastapi import APIRouter, Depends, HTTPException

from app.api.limits import enforce_research_rate_limit
from app.api.schemas import (
    ResearchQueryRequest,
    ResearchQueryResponse,
    ResearchSourceResponse,
)
from app.config import settings
from app.ingestion.paper_service import normalize_arxiv_id
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.embeddings import SentenceTransformerEmbedder
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.retriever import VectorRetriever
from app.retrieval.vector_store import DEFAULT_INDEX_PATH, FaissVectorStore
from app.services.llm_service import (
    LLMConfigurationError,
    LLMProviderUnavailableError,
    LLMQuotaError,
    LLMRequestError,
    LLMTimeoutError,
    create_llm_service,
)
from app.services.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/research", tags=["research"])


@contextmanager
def _log_stage(stage: str, **fields: object) -> Iterator[None]:
    """Log the duration of a pipeline stage."""
    started = perf_counter()
    details = " ".join(
        f"{name}={value}" for name, value in fields.items()
    )

    logger.info("research_stage_started stage=%s %s", stage, details)

    try:
        yield
    except Exception as error:
        logger.warning(
            "research_stage_failed stage=%s elapsed_seconds=%.3f error_type=%s",
            stage,
            perf_counter() - started,
            type(error).__name__,
        )
        raise
    else:
        logger.info(
            "research_stage_completed stage=%s elapsed_seconds=%.3f %s",
            stage,
            perf_counter() - started,
            details,
        )


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    """Load the existing FAISS index and construct the RAG pipeline."""

    if not DEFAULT_INDEX_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "The prebuilt FAISS index is missing. Deploy both "
                "data/index/researchiq.faiss and "
                "data/index/researchiq.json."
            ),
        )

    try:
        vector_store = FaissVectorStore.load(DEFAULT_INDEX_PATH)
    except (FileNotFoundError, ValueError, OSError) as error:
        logger.exception("Could not load the prebuilt FAISS index.")
        raise HTTPException(
            status_code=503,
            detail=(
                "The FAISS index or its metadata file is "
                "invalid or unavailable."
            ),
        ) from error

    dense_retriever = VectorRetriever(
        vector_store,
        SentenceTransformerEmbedder(),
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=BM25Retriever(vector_store.chunks),
        reranker=None,
        dense_top_k=settings.dense_top_k,
        bm25_top_k=settings.bm25_top_k,
        rrf_k=settings.rrf_k,
        rerank_top_k=settings.rerank_top_k,
        final_top_k=settings.final_top_k,
    )

    return RAGPipeline(
        retriever=hybrid_retriever,
        llm=create_llm_service(),
    )


@router.post(
    "/query",
    response_model=ResearchQueryResponse,
    dependencies=[Depends(enforce_research_rate_limit)],
)
def research_query(
    request: ResearchQueryRequest,
) -> ResearchQueryResponse:
    """Answer a question using the prebuilt research index."""

    with _log_stage("retrieval_pipeline_load"):
        pipeline = get_rag_pipeline()

    try:
        with _log_stage(
            "retrieval_and_generation",
            top_k=request.top_k,
        ):
            result = pipeline.answer_question(
                request.question,
                top_k=request.top_k,
            )

    except LLMConfigurationError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except LLMProviderUnavailableError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except LLMTimeoutError as error:
        raise HTTPException(
            status_code=504,
            detail=str(error),
        ) from error

    except LLMQuotaError as error:
        raise HTTPException(
            status_code=429,
            detail=str(error),
        ) from error

    except LLMRequestError as error:
        raise HTTPException(
            status_code=502,
            detail=str(error),
        ) from error

    if not result.sources:
        result = type(result)(
            answer=(
                "I could not find relevant academic evidence in the "
                "prebuilt indexed papers, so I cannot answer this "
                "question reliably. The deployed demo uses a fixed "
                "corpus and does not fetch new arXiv papers per query."
            ),
            sources=[],
        )

    return ResearchQueryResponse(
        answer=result.answer,
        sources=[
            ResearchSourceResponse(
                citation_id=f"[{source.citation_id}]",
                chunk_id=source.chunk_id,
                paper_id=source.paper_id,
                paper_url=_paper_url(source.paper_id),
                title=source.title,
                authors=source.authors,
                section=source.section,
                page=source.page,
                text=source.text,
                similarity_score=source.similarity_score,
                retrieval_score=source.retrieval_score,
                retrieval_method=source.retrieval_method,
            )
            for source in result.sources
        ],
    )


def _paper_url(paper_id: str) -> str | None:
    try:
        normalized_id = normalize_arxiv_id(paper_id)
    except ValueError:
        return None

    return f"https://arxiv.org/abs/{normalized_id}"