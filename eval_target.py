from langsmith import traceable
from workflow import ResearchWorkflow



@traceable(name="research_assistant_target")
def research_assistant_target(inputs: dict) -> dict:
    """
    Accepts flexible dataset schemas.
    """

    # Accept common dataset field names
    query = (
        inputs.get("query")
        or inputs.get("input")
        or inputs.get("question")
        or inputs.get("research_request") 
    )

    if not query:
        raise ValueError(
            f"Dataset example inputs must contain one of "
            f"['query', 'input', 'question'], got: {list(inputs.keys())}"
        )

    workflow = ResearchWorkflow(
        skip_formatting=False,
        enable_tools=False,
        detailed_output=False
    )

    output = workflow.run(query)

    return {"output": output}
