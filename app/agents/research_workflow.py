"""Bounded LangGraph orchestration for research question analysis."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.embeddings import SentenceTransformerEmbedder
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import VectorRetriever
from app.retrieval.vector_store import DEFAULT_INDEX_PATH, FaissVectorStore
from app.services.llm_service import create_llm_service
from app.services.rag_pipeline import RAGPipeline


class ResearchWorkflowState(TypedDict):
    """Typed state for the bounded research workflow."""

    original_question: str
    research_plan: list[str]
    retrieved_evidence: list[dict[str, Any]]
    draft_answer: str
    citations: list[str]
    critique: dict[str, Any]
    iteration: int
    final_answer: str


class ResearchWorkflow:
    """Use a bounded planner → researcher → critic loop to produce grounded answers."""

    def __init__(
        self,
        retriever: Any | None = None,
        llm: Any | None = None,
        *,
        max_iterations: int = 2,
    ) -> None:
        self.retriever = retriever or self._build_default_retriever()
        self.llm = llm or create_llm_service()
        self.max_iterations = max_iterations
        self._graph = self.build_graph()

    @staticmethod
    def _build_default_retriever() -> Any:
        try:
            vector_store = FaissVectorStore.load(DEFAULT_INDEX_PATH)
        except (FileNotFoundError, ValueError, OSError) as error:
            raise RuntimeError(
                "The local FAISS retrieval index is unavailable. Run scripts/build_index.py first."
            ) from error

        dense_retriever = VectorRetriever(vector_store, SentenceTransformerEmbedder())
        bm25_retriever = BM25Retriever(vector_store.chunks)
        return HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            reranker=CrossEncoderReranker(),
            dense_top_k=settings.dense_top_k,
            bm25_top_k=settings.bm25_top_k,
            rrf_k=settings.rrf_k,
            rerank_top_k=settings.rerank_top_k,
            final_top_k=settings.final_top_k,
        )

    def planner(self, state: ResearchWorkflowState) -> ResearchWorkflowState:
        question = str(state["original_question"]).strip()
        if not question:
            raise ValueError("Research question must not be empty.")

        plan = [
            "Identify the specific research question and required evidence scope.",
            "Retrieve the most relevant passages using the existing hybrid retrieval stack.",
            "Draft an evidence-grounded answer and check whether each claim is supported by citations.",
        ]
        state["research_plan"] = plan
        return state

    def researcher(self, state: ResearchWorkflowState) -> ResearchWorkflowState:
        question = str(state["original_question"]).strip()
        if not question:
            raise ValueError("Research question must not be empty.")

        pipeline = RAGPipeline(retriever=self.retriever, llm=self.llm)
        result = pipeline.answer_question(question, top_k=settings.final_top_k)

        state["retrieved_evidence"] = [
            {
                "citation_id": source.citation_id,
                "chunk_id": source.chunk_id,
                "paper_id": source.paper_id,
                "title": source.title,
                "authors": source.authors,
                "section": source.section,
                "page": source.page,
                "text": source.text,
                "similarity_score": source.similarity_score,
                "retrieval_score": source.retrieval_score,
                "retrieval_method": source.retrieval_method,
            }
            for source in result.sources
        ]
        state["draft_answer"] = result.answer
        state["citations"] = [str(source.citation_id) for source in result.sources]
        state["iteration"] = int(state.get("iteration", 0)) + 1
        state["critique"] = {"accepted": False, "feedback": "Pending critique."}
        return state

    def critic(self, state: ResearchWorkflowState) -> ResearchWorkflowState:
        answer = str(state.get("draft_answer", "")).strip()
        evidence = state.get("retrieved_evidence", [])
        citations = [str(citation) for citation in state.get("citations", [])]
        bracketed = re.findall(r"\[(\d+)\]", answer)
        unsupported = [value for value in bracketed if value not in citations]

        if not evidence:
            critique = {
                "accepted": False,
                "feedback": "No evidence was retrieved for this question; the answer cannot be grounded yet.",
            }
        elif not answer:
            critique = {
                "accepted": False,
                "feedback": "The draft answer is empty and cannot be evaluated for evidence support.",
            }
        elif not bracketed and citations:
            critique = {
                "accepted": False,
                "feedback": "The draft answer did not include citations for the retrieved evidence.",
            }
        elif unsupported:
            critique = {
                "accepted": False,
                "feedback": "The answer references unsupported citation labels: " + ", ".join(unsupported),
            }
        else:
            critique = {
                "accepted": True,
                "feedback": "The draft answer is sufficiently grounded in the retrieved evidence.",
            }
            state["final_answer"] = answer

        state["critique"] = critique
        return state

    def should_continue(self, state: ResearchWorkflowState) -> Literal["researcher", "__end__"]:
        critique = state.get("critique", {})
        accepted = bool(critique.get("accepted", False))
        iteration = int(state.get("iteration", 0))
        if accepted:
            return "__end__"
        if iteration < self.max_iterations:
            return "researcher"
        return "__end__"

    def build_graph(self) -> StateGraph:
        workflow = StateGraph(ResearchWorkflowState)
        workflow.add_node("planner", self.planner)
        workflow.add_node("researcher", self.researcher)
        workflow.add_node("critic", self.critic)
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "researcher")
        workflow.add_edge("researcher", "critic")
        workflow.add_conditional_edges(
            "critic",
            self.should_continue,
            {
                "researcher": "researcher",
                "__end__": END,
            },
        )
        return workflow.compile()

    def run(self, question: str) -> ResearchWorkflowState:
        state: ResearchWorkflowState = {
            "original_question": question,
            "research_plan": [],
            "retrieved_evidence": [],
            "draft_answer": "",
            "citations": [],
            "critique": {"accepted": False, "feedback": "Not yet reviewed."},
            "iteration": 0,
            "final_answer": "",
        }
        result = self._graph.invoke(state)
        if not result.get("final_answer") and result.get("draft_answer"):
            result["final_answer"] = result["draft_answer"]
        return result
