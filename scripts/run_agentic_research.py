"""Run the bounded LangGraph research workflow for a single question."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.research_workflow import ResearchWorkflow


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the bounded ResearchIQ agentic workflow.")
    parser.add_argument("question", help="Research question to answer with the bounded workflow.")
    arguments = parser.parse_args()

    try:
        workflow = ResearchWorkflow()
        result = workflow.run(arguments.question)
    except RuntimeError as error:
        print(f"Workflow could not start: {error}", file=sys.stderr)
        return 1

    print("Research plan:")
    for step in result.get("research_plan", []):
        print(f"- {step}")

    print("\nFinal answer:")
    print(result.get("final_answer", result.get("draft_answer", "No grounded answer available.")))

    print("\nCitations:")
    for citation in result.get("citations", []):
        print(f"- {citation}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
