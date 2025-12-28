"""Enhanced LangGraph workflow for agentic research assistance with conditional edges."""
from typing import TypedDict, List, Dict, Optional, Literal
from langgraph.graph import StateGraph, END
from search_service import GoogleSearchService
from llm_service import OllamaLLMService
from langsmith import traceable
import re


class ResearchState(TypedDict):
    """Enhanced state object for the research workflow with forward compatibility."""

    # Core workflow fields
    original_query: str
    refined_query: str
    search_results: List[Dict[str, str]]
    formatted_results: str
    summary: str
    output: str

    # Task 1 integration - structured outputs from prompt templates
    intent_metadata: Dict          # From intent_parser
    evaluation_result: Dict        # From search_evaluator (drives conditional routing)
    validated_facts: Dict          # From reasoning
    ranked_documents: Dict         # From retrieval_ranker

    # Loop protection
    iteration_count: int           # Track refine-and-research iterations
    max_iterations: int            # Maximum loops allowed (default: 3)

    # Task 3 placeholders (tools integration)
    vector_db_results: Optional[List[Dict]]       # For vector database results
    retrieved_documents: Optional[List[Dict]]     # For document retriever
    tool_outputs: Optional[Dict]                  # Generic tool outputs

    # Task 4 placeholders (evaluation framework)
    metrics: Dict                  # Performance metrics (tokens, latency, quality)
    evaluation_metadata: Dict      # Tracking data for evaluation


class ResearchWorkflow:
    """Enhanced LangGraph workflow with agentic behavior and conditional routing."""

    def __init__(self, skip_formatting: bool = False, enable_tools: bool = False,
                 cached_results_count: int = 3, local_docs_count: int = 2,
                 local_docs_score_threshold: float = 0.5,
                 detailed_output: bool = False):
        """
        Initialize the research workflow.

        Args:
            skip_formatting: If True, skips the format_output node and returns raw summary.
                           This speeds up execution by skipping one LLM call.
            enable_tools: If True, enables tool calling (vector search, document loader).
                        Tools are cached in ChromaDB and available for semantic search.
            cached_results_count: Number of cached results to retrieve from ChromaDB (default: 3).
                                Only used when enable_tools is True.
            local_docs_count: Number of local documents to retrieve (default: 2).
                            Only used when enable_tools is True.
            local_docs_score_threshold: Relevance threshold for local documents (default: 0.5).
                                      Lower = stricter filtering. Only docs with L2 distance <= threshold
                                      are included. Set to 0 to disable filtering (include all).
                                      Typical values: 0.3 (strict), 0.5 (moderate), 2.0 (lenient).
                                      Only used when enable_tools is True.
            detailed_output: If True, includes validated facts with sources in final output.
                           Useful for research/demo purposes to show full transparency.
        """
        self.search_service = GoogleSearchService()
        self.llm_service = OllamaLLMService()
        self.skip_formatting = skip_formatting
        self.enable_tools = enable_tools
        self.cached_results_count = cached_results_count
        self.local_docs_count = local_docs_count
        self.local_docs_score_threshold = local_docs_score_threshold
        self.detailed_output = detailed_output
        self.local_docs_indexed = False  # Track if we've indexed local docs this session
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the enhanced LangGraph workflow graph with conditional edges."""
        workflow = StateGraph(ResearchState)

        # Add all nodes
        workflow.add_node("parse_intent", self._parse_intent_node)
        workflow.add_node("refine_query", self._refine_query_node)
        workflow.add_node("fetch_sources", self._fetch_sources_node)
        workflow.add_node("evaluate_search", self._evaluate_search_node)
        workflow.add_node("rank_documents", self._rank_documents_node)
        workflow.add_node("reason_and_synthesize", self._reason_and_synthesize_node)
        workflow.add_node("summarize", self._summarize_node)

        # Conditionally add format_output node
        if not self.skip_formatting:
            workflow.add_node("format_output", self._format_output_node)

        # Define the workflow edges
        workflow.set_entry_point("parse_intent")
        workflow.add_edge("parse_intent", "refine_query")
        workflow.add_edge("refine_query", "fetch_sources")
        workflow.add_edge("fetch_sources", "evaluate_search")

        # CONDITIONAL EDGE: Routing after search evaluation
        # Build mapping based on skip_formatting setting
        evaluation_routes = {
            "rank_documents": "rank_documents",          # Results are good
            "refine_query": "refine_query",              # Loop back to refine
        }

        # Note: We changed max_iterations routing to go to rank_documents instead of format_output,
        # so format_output is no longer used in routing. Only include it if the node exists.
        if not self.skip_formatting:
            evaluation_routes["format_output"] = "format_output"

        workflow.add_conditional_edges(
            "evaluate_search",
            self._route_after_evaluation,
            evaluation_routes
        )

        # Continue linear flow after ranking
        workflow.add_edge("rank_documents", "reason_and_synthesize")
        workflow.add_edge("reason_and_synthesize", "summarize")

        # Conditionally add formatting step
        if self.skip_formatting:
            # Skip formatting - go directly from summarize to END
            workflow.add_edge("summarize", END)
        else:
            # Include formatting step
            workflow.add_edge("summarize", "format_output")
            workflow.add_edge("format_output", END)

        return workflow.compile()

    # ========================================================================
    # NODE IMPLEMENTATIONS
    # ========================================================================

    def _parse_intent_node(self, state: ResearchState) -> ResearchState:
        """Node: Parse user query to extract intent and metadata."""
        print(f"🎯 Parsing intent from query...")
        intent_metadata = self.llm_service.parse_intent(state['original_query'])
        state['intent_metadata'] = intent_metadata

        # Extract facts count for later use
        num_facts = intent_metadata.get('required_facts_count', 3)
        print(f"✅ Intent: {intent_metadata.get('primary_intent', 'unknown')}, "
              f"Facts needed: {num_facts}")
        return state

    def _refine_query_node(self, state: ResearchState) -> ResearchState:
        """Node: Refine the user's query for better search results."""
        iteration = state.get('iteration_count', 0)

        # Increment iteration counter if this is a loop-back (evaluation_result exists)
        if state.get('evaluation_result'):
            iteration += 1
            state['iteration_count'] = iteration

        if iteration > 0:
            print(f"🔄 Refining query (iteration {iteration})...")
        else:
            print(f"🔍 Refining query...")

        # Get evaluation feedback if available (for loops)
        evaluation_feedback = state.get('evaluation_result')

        refined = self.llm_service.refine_query(
            state['original_query'],
            intent_metadata=state.get('intent_metadata'),
            evaluation_feedback=evaluation_feedback
        )
        state['refined_query'] = refined
        print(f"✅ Refined query: {refined}")
        return state

    def _fetch_sources_node(self, state: ResearchState) -> ResearchState:
        """Node: Fetch information from web sources."""

        cached_results = []
        local_docs = []

        if self.enable_tools:
            from vector_store_service import get_vector_store_service
            vector_service = get_vector_store_service()

            # Index local documents on first call
            if not self.local_docs_indexed:
                print(f"📁 Indexing local documents...")
                new_docs = vector_service.index_local_documents(
                    documents_dir="./documents",
                    verbose=True
                )
                self.local_docs_indexed = True
                if new_docs == 0:
                    print(f"ℹ️  No new documents to index")

            # Query ChromaDB for cached web results
            print(f"🔍 Querying ChromaDB for cached results...")
            cached_results = vector_service.similarity_search(
                state['refined_query'],
                top_k=self.cached_results_count,
                score_threshold=self.local_docs_score_threshold,
                verbose=True
            )

            if cached_results:
                print(f"✅ Found {len(cached_results)} relevant cached results")
            else:
                print(f"ℹ️  No relevant cached results found")

            # Query for relevant local documents
            print(f"📄 Querying local documents...")
            local_docs = vector_service.search_local_documents(
                state['refined_query'],
                top_k=self.local_docs_count,
                score_threshold=self.local_docs_score_threshold,
                verbose=True
            )

            if local_docs:
                print(f"✅ Found {len(local_docs)} relevant local documents")
            else:
                print(f"ℹ️  No relevant local documents found")

        # Fetch fresh web results
        print(f"🌐 Searching the web...")
        results = self.search_service.search(state['refined_query'], num_results=5)
        state['search_results'] = results

        # Store retrieved documents for source citations
        state['retrieved_documents'] = local_docs if local_docs else []

        # Format results for LLM - include fresh, cached, and local
        formatted = self.search_service.format_results_for_llm(results)

        # Add cached results if available
        if cached_results:
            cached_formatted = "\n\n--- CACHED RESULTS FROM PREVIOUS RESEARCH ---\n"
            for i, cached in enumerate(cached_results, 1):
                title = cached.get('title', 'Untitled')
                cached_formatted += f"\n[CACHED] {title}:\n"
                cached_formatted += f"Content: {cached.get('content', 'N/A')}\n"
                if cached.get('link'):
                    cached_formatted += f"Source: {cached['link']}\n"
            formatted = formatted + cached_formatted

        # Add local documents if available
        if local_docs:
            local_formatted = "\n\n--- LOCAL REFERENCE DOCUMENTS ---\n"
            for i, doc in enumerate(local_docs, 1):
                filename = doc.get('title', 'Untitled')
                local_formatted += f"\n[LOCAL DOC] {filename}:\n"
                local_formatted += f"Content: {doc.get('content', 'N/A')}\n"
            formatted = formatted + local_formatted

        state['formatted_results'] = formatted
        print(f"✅ Found {len(results)} fresh search results")

        # Cache new results in vector store if tools are enabled
        if self.enable_tools:
            from vector_store_service import get_vector_store_service
            vector_service = get_vector_store_service()
            print(f"📦 Caching new results to ChromaDB...")
            vector_service.add_search_results(results, verbose=True)

        return state

    def _evaluate_search_node(self, state: ResearchState) -> ResearchState:
        """Node: Evaluate if search results are sufficient."""
        print(f"📊 Evaluating search results quality...")

        required_facts = state.get('intent_metadata', {}).get('required_facts_count', 3)
        evaluation = self.llm_service.evaluate_search_results(
            state['original_query'],
            state['formatted_results'],
            required_facts_count=required_facts
        )

        state['evaluation_result'] = evaluation

        recommendation = evaluation.get('recommendation', 'proceed')
        relevance = evaluation.get('relevance_score', 0)
        print(f"✅ Evaluation: {recommendation} (relevance: {relevance}/10)")

        return state

    def _rank_documents_node(self, state: ResearchState) -> ResearchState:
        """Node: Rank documents by relevance."""
        print(f"📋 Ranking documents by relevance...")

        primary_intent = state.get('intent_metadata', {}).get('primary_intent', 'unknown')
        ranking = self.llm_service.rank_documents(
            state['original_query'],
            primary_intent,
            state['formatted_results']
        )

        state['ranked_documents'] = ranking
        num_ranked = len(ranking.get('ranked_documents', []))
        print(f"✅ Ranked {num_ranked} documents")
        return state

    def _reason_and_synthesize_node(self, state: ResearchState) -> ResearchState:
        """Node: Analyze and synthesize information from multiple sources."""
        print(f"🧠 Reasoning and synthesizing information...")

        required_facts = state.get('intent_metadata', {}).get('required_facts_count', 3)
        reasoning = self.llm_service.reason_and_synthesize(
            state['original_query'],
            state['formatted_results'],
            required_facts_count=required_facts
        )

        state['validated_facts'] = reasoning
        num_facts = len(reasoning.get('validated_facts', []))
        print(f"✅ Validated {num_facts} facts")
        return state

    def _summarize_node(self, state: ResearchState) -> ResearchState:
        """Node: Summarize validated facts into structured output."""
        print(f"📝 Creating summary...")

        required_facts = state.get('intent_metadata', {}).get('required_facts_count', 3)
        summary = self.llm_service.summarize_results(
            state['original_query'],
            state.get('validated_facts', {}),
            state['formatted_results'],
            num_facts=required_facts
        )

        state['summary'] = summary

        # If skipping formatting, set output to summary directly
        if self.skip_formatting:
            state['output'] = summary
            print(f"✅ Summary generated (using as final output - formatting skipped)")
        else:
            print(f"✅ Summary generated")

        return state

    def _format_output_node(self, state: ResearchState) -> ResearchState:
        """Node: Format the final research report."""
        print(f"🎨 Formatting final output...")

        # Combine web sources and local documents for citations
        all_sources = state.get('search_results', []).copy()
        local_docs = state.get('retrieved_documents', [])

        # Add local docs as sources with appropriate formatting
        for doc in local_docs:
            all_sources.append({
                'title': f"Local Document: {doc.get('title', 'Untitled')}",
                'link': doc.get('file_path', 'Local file')
            })

        # Include validated facts if detailed output is enabled
        validated_facts = state.get('validated_facts', {}) if self.detailed_output else None

        output = self.llm_service.format_output(
            state['original_query'],
            state['refined_query'],
            state['summary'],
            all_sources,
            validated_facts=validated_facts
        )

        state['output'] = output
        print(f"✅ Output formatted")
        return state

    # ========================================================================
    # CONDITIONAL EDGE ROUTING FUNCTIONS
    # ========================================================================

    def _route_after_evaluation(
        self,
        state: ResearchState
    ) -> Literal["rank_documents", "refine_query", "format_output"]:
        """
        Conditional routing after search evaluation.

        Implements the refine-and-research loop with protection against infinite loops.

        Returns:
            Next node name to route to
        """
        evaluation = state.get('evaluation_result', {})
        recommendation = evaluation.get('recommendation', 'proceed')
        iteration_count = state.get('iteration_count', 0)
        max_iterations = state.get('max_iterations', 3)

        # Loop protection: Prevent infinite refine-and-research cycles
        if iteration_count >= max_iterations:
            print(f"⚠️ Max iterations ({max_iterations}) reached. Proceeding with current results.")
            return "rank_documents"  # Stop looping, but continue with analysis and synthesis

        # Route based on evaluation recommendation
        if recommendation == "proceed":
            print(f"✅ Search results are sufficient. Continuing workflow...")
            return "rank_documents"

        elif recommendation == "refine_and_research":
            gaps = evaluation.get('identified_gaps', [])
            print(f"🔄 Search results insufficient (gaps: {gaps}). Refining and re-searching...")
            return "refine_query"  # Loop back

        else:
            # Fallback: treat unknown recommendations as "proceed"
            print(f"⚠️ Unknown recommendation '{recommendation}'. Proceeding...")
            return "rank_documents"

    # ========================================================================
    # WORKFLOW EXECUTION
    # ========================================================================
    @traceable(name="research_workflow")
    def run(self, research_request: str) -> str:
        """
        Execute the enhanced research workflow.

        Args:
            research_request: Natural language research request

        Returns:
            Formatted research results
        """
        # Initialize state with all required fields
        initial_state: ResearchState = {
            # Core fields
            "original_query": research_request,
            "refined_query": "",
            "search_results": [],
            "formatted_results": "",
            "summary": "",
            "output": "",

            # Task 1 integration
            "intent_metadata": {},
            "evaluation_result": {},
            "validated_facts": {},
            "ranked_documents": {},

            # Loop protection
            "iteration_count": 0,
            "max_iterations": 3,

            # Task 3 placeholders
            "vector_db_results": None,
            "retrieved_documents": None,
            "tool_outputs": None,

            # Task 4 placeholders
            "metrics": {},
            "evaluation_metadata": {}
        }

        print(f"\n{'='*60}")
        print(f"🚀 Starting Enhanced Research Workflow")
        print(f"{'='*60}\n")

        final_state = self.graph.invoke(initial_state)

        print(f"\n{'='*60}")
        print(f"✅ Workflow Complete!")
        print(f"{'='*60}\n")

        return final_state['output']
