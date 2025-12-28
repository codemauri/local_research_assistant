"""
Run LangSmith evaluation for the local research assistant.

- Compatible with LangSmith 0.5.1
- Assumes this file is in the PROJECT ROOT
- Dataset already exists in LangSmith UI
- Uses local Ollama as LLM-as-judge
"""

from langsmith.evaluation import evaluate
from langchain_ollama import ChatOllama

from eval_target import research_assistant_target
from openevals.llm import create_llm_as_judge
from openevals.prompts import CONCISENESS_PROMPT


# ============================================================
# LOCAL LLM (OLLAMA) USED AS JUDGE
# ============================================================

judge_llm = ChatOllama(
    model="llama3.2",
    temperature=0.0,
)


conciseness_evaluator = create_llm_as_judge(
    prompt=CONCISENESS_PROMPT,
    feedback_key="conciseness",
    model="ollama:llama3.2",
)


# ============================================================
# HELPER: EXTRACT QUERY SAFELY FROM DATASET EXAMPLE
# ============================================================

def get_query(example):
    """
    Accepts flexible dataset schemas.
    """
    return (
        example.inputs.get("query")
        or example.inputs.get("input")
        or example.inputs.get("question")
        or example.inputs.get("research_request")
        or ""
    )


# ============================================================
# LLM-AS-JUDGE EVALUATOR
# ============================================================

def research_quality_judge(run, example):
    query = get_query(example)
    answer = run.outputs.get("output") or ""

    if not answer.strip():
        return {
            "score": 0,
            "comment": "No output produced by the target function.",
        }

    prompt = f"""
You are evaluating a research assistant answer.

Question:
{query}

Answer:
{answer}

Score each criterion from 1 (poor) to 5 (excellent):

- Accuracy (facts correct, no hallucinations)
- Coverage (addresses key aspects of the question)
- Relevance (stays on topic)
- Coherence (logical structure and flow)
- Clarity (clear and precise language)

Return ONLY plain text with short justifications.
"""

    result = judge_llm.invoke(prompt).content

    return {
        "comment": result,
    }


# ============================================================
# DETERMINISTIC / HEURISTIC EVALUATORS
# ============================================================

def minimum_length_eval(run, example):
    output = run.outputs.get("output") or ""
    word_count = len(output.split())

    return {
        "score": 1.0 if word_count >= 120 else 0.0,
        "word_count": word_count,
        "comment": "Minimum length requirement (120 words)",
    }


def mentions_sources_eval(run, example):
    output = (run.outputs.get("output") or "").lower()

    has_sources = any(
        token in output
        for token in [
            "http",
            "https",
            "according to",
            "study",
            "report",
            "source",
            "research",
        ]
    )

    return {
        "score": 1.0 if has_sources else 0.0,
        "comment": "Mentions evidence or sources",
    }


# ============================================================
# RUN DATASET EVALUATION
# ============================================================

if __name__ == "__main__":
    evaluate(
        research_assistant_target,          # TARGET (positional)
        data="local_research_assistant_evaluator",       # Dataset name from LangSmith UI
        evaluators=[
            research_quality_judge,          # LLM-as-judge
            minimum_length_eval,             # Deterministic guardrail
            mentions_sources_eval,            # Heuristic groundedness
            conciseness_evaluator,           # pre-built
        ],
    )
