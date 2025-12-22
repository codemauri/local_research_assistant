Building and Evaluating an Agentic AI System with LangGraph
✔ Demonstrate agentic behavior (goal planning, tool usage, iterative reasoning).
✔ Use prompt engineering to guide model actions effectively.
✔ Leverage LangGraph for flexible orchestration and workflow state management.
✔ Include an evaluation framework to assess the agent's performance quantitatively and qualitatively.


Prompt Engineering
Describe prompt engineering — the art of crafting and structuring prompts to get useful, consistent, and accurate outputs from LLM-based systems. Discuss techniques like clear instruction structures, context provision, and templating.
        Limitation from Prompt engineering: 1. Highly relies on the model capability. 2. Ambiguity & unpredictability. 3. Lack of true understanding.
How can we improve prompt engineering? 1. Finetuning model 2. RAG (retrieval augmented generation) 3. Post training(RLHF, DPO, …)
We will focus on RAG today.


Agentic AI
Explain what agentic AI means: AI systems that behave autonomously, plan multi-step workflows, and interact with tools and data to achieve goals rather than simply respond to prompts. Provide examples (e.g., research agents, workflow assistants).
LangGraph
Define LangGraph as a graph-based orchestration framework that lets developers define workflows, state transitions, and multi-agent coordination with nodes and edges instead of linear chains. https://www.langchain.com/langgraph
Evaluation Frameworks
Overview techniques for evaluating AI agents: correctness, relevance, coherence, safety, and other task-specific metrics. You can use LLM-as-a-judge methods where a model scores its own output according to criteria.

________________


Project Tasks
Task 1: Prompt Engineering
Create templates for each workflow step (e.g., search, plan, summarize).
Example Prompt Template Structure:
SYSTEM: You are an autonomous research agent.
TASK: {TaskDescription}
CONTEXT: {Relevant Context}
OUTPUT FORMAT: {JSON schema or structured text}


Explain why you chose this structure based on prompt engineering best practices. Medium
________________


Task 2: Build the Agent with LangGraph
* Define nodes for intent parsing, web search, retrieval, reasoning, summarization, and output formatting.

* Define edges with conditions (e.g., if data is insufficient, re-search or ask clarifying questions).

* Maintain shared state across steps.

Include:
   * Graph definition code

   * Node implementation samples

   * State schema

(Note: LangGraph's graph API makes workflows easier to express than linear chains.) LangChain Blog
________________


Task 3: Integrate Tools
      * A search API (e.g., SerpAPI or Bing)
      * A document retriever
      * A vector database

      * …
Explain how these tools improve agent capabilities beyond isolated LLM calls.
________________


Task 4: Evaluation Framework
Design and implement an evaluation pipeline for your agent:
Quantitative Metrics
         * Accuracy: Compare agent output to ground truth (if available).
         * Relevance: Score relevance with an LLM judge or rubric.
         * Coherence: Fluency and logical flow.

Qualitative Evaluation
            * Human review of responses
            * Error analysis (what errors occur, why?)

Produce a dashboard or report summarizing results.

________________


Bonus Extensions
Development level
               1. Add clarification loops when the agent isn't confident.
               2. Track runtime metrics (tokens used, latency).
               3. Deploy as a web UI with chat interface.
               1. Streamlit;
               4. Connect with a private vector database.
Application level
               1. Can you create an Agent that provides investment suggestions based on the famous investors' style? (e.g., Warren Buffett, Ray Dalio, …)
               1. Use the investment principle as a guideline
               2. Can you connect this development pipeline with your domain expertise or daily job?




https://github.com/danielxu05/local_research_assistant/tree/main






Appendix


LLM stages
Pretrain
Finetuning
Post training
        Response 1 >> response 2


Question answering task:
        Question -> answer
RAG
        Question -> collect some related information(docs) -> LLM generate answers
