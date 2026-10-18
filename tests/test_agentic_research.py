from __future__ import annotations

from app.agents.research_workflow import ResearchWorkflow
from app.retrieval.retriever import RetrievedChunk


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self.chunks[:top_k]


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def _make_chunk(chunk_id: str, text: str, *, page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        paper_id="1706.03762v7",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani"],
        section="Introduction",
        page=page,
        text=text,
        similarity_score=0.95,
        retrieval_method="hybrid_rrf_reranked",
    )


def test_planner_generates_structured_research_plan() -> None:
    workflow = ResearchWorkflow(
        retriever=FakeRetriever([_make_chunk("chunk-a", "Attention is the main contribution.")]),
        llm=FakeLLM(["The main contribution is attention [1]."]),
    )

    state = workflow.planner({"original_question": "What is the main contribution of the model?"})

    assert state["research_plan"][0].startswith("Identify")
    assert any("evidence" in item.lower() for item in state["research_plan"])


def test_researcher_output_keeps_citations_and_metadata() -> None:
    chunk = _make_chunk("chunk-a", "The model uses self-attention instead of recurrence.")
    workflow = ResearchWorkflow(retriever=FakeRetriever([chunk]), llm=FakeLLM(["The main contribution is attention [1]."]))

    state = workflow.researcher({
        "original_question": "What is the main contribution of the model?",
        "research_plan": ["Plan"],
        "retrieved_evidence": [],
        "draft_answer": "",
        "citations": [],
        "critique": {"accepted": False, "feedback": ""},
        "iteration": 0,
        "final_answer": "",
    })

    assert state["draft_answer"] == "The main contribution is attention [1]."
    assert state["citations"] == ["1"]
    assert state["retrieved_evidence"][0]["chunk_id"] == "chunk-a"
    assert state["iteration"] == 1


def test_critic_accepts_well_supported_answer() -> None:
    workflow = ResearchWorkflow(retriever=FakeRetriever([]), llm=FakeLLM([]))
    state = {
        "original_question": "What is the main contribution?",
        "research_plan": ["Plan"],
        "retrieved_evidence": [{"citation_id": 1, "chunk_id": "chunk-a"}],
        "draft_answer": "The model uses attention [1].",
        "citations": ["1"],
        "critique": {"accepted": False, "feedback": ""},
        "iteration": 1,
        "final_answer": "",
    }

    result = workflow.critic(state)

    assert result["critique"]["accepted"] is True
    assert result["final_answer"] == "The model uses attention [1]."


def test_critic_rejection_routes_to_one_correction_pass() -> None:
    workflow = ResearchWorkflow(retriever=FakeRetriever([_make_chunk("chunk-a", "Attention is key.")]), llm=FakeLLM(["Unsupported claim without citation.", "The model uses attention [1]."]))

    graph = workflow.build_graph()
    result = graph.invoke({
        "original_question": "What is the main contribution?",
        "research_plan": [],
        "retrieved_evidence": [],
        "draft_answer": "",
        "citations": [],
        "critique": {"accepted": False, "feedback": ""},
        "iteration": 0,
        "final_answer": "",
    })

    assert result["final_answer"] == "The model uses attention [1]."
    assert result["iteration"] == 2
    assert result["critique"]["accepted"] is True


def test_maximum_iteration_limit_stops_after_two_research_cycles() -> None:
    workflow = ResearchWorkflow(retriever=FakeRetriever([]), llm=FakeLLM(["No citation [99]."]))

    result = workflow.should_continue({
        "original_question": "Why?",
        "research_plan": [],
        "retrieved_evidence": [],
        "draft_answer": "No citation [99].",
        "citations": ["1"],
        "critique": {"accepted": False, "feedback": "bad"},
        "iteration": 2,
        "final_answer": "",
    })

    assert result == "__end__"


def test_citation_preservation_in_state() -> None:
    chunk = _make_chunk("chunk-b", "Self-attention computes long-range dependencies.")
    workflow = ResearchWorkflow(retriever=FakeRetriever([chunk]), llm=FakeLLM(["Self-attention captures dependencies [1]."]))

    result = workflow.run("How does self-attention work?")

    assert "chunk-b" == result["retrieved_evidence"][0]["chunk_id"]
    assert "1" in result["citations"]
    assert "[1]" in result["final_answer"]
