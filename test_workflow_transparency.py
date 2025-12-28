"""
Workflow Transparency Test
===========================

This test file provides complete visibility into the workflow execution.
It displays the output of each node and the exact data passed between nodes.

Usage:
    python test_workflow_transparency.py "your query here"
"""

import sys
import json
import time
from typing import Dict, Any, List, Tuple
from workflow import ResearchWorkflow, ResearchState


class WorkflowDebugger:
    """Wrapper that provides transparent visibility into workflow execution."""

    def __init__(
        self,
        skip_formatting: bool = False,
        enable_tools: bool = False,
        cached_results_count: int = 3,
        local_docs_count: int = 2,
        local_docs_score_threshold: float = 0.5,
        detailed_output: bool = False
    ):
        self.workflow = ResearchWorkflow(
            skip_formatting=skip_formatting,
            enable_tools=enable_tools,
            cached_results_count=cached_results_count,
            local_docs_count=local_docs_count,
            local_docs_score_threshold=local_docs_score_threshold,
            detailed_output=detailed_output
        )
        self.step_count = 0
        self.node_timings: List[Tuple[str, float, int]] = []  # (node_name, duration, call_number)
        self.node_call_counts: Dict[str, int] = {}  # Track how many times each node is called
        self.workflow_start_time = None
        self.workflow_end_time = None
        self.skip_formatting = skip_formatting
        self.enable_tools = enable_tools
        self.cached_results_count = cached_results_count
        self.local_docs_count = local_docs_count
        self.local_docs_score_threshold = local_docs_score_threshold
        self.detailed_output = detailed_output

    def format_state_field(self, field_name: str, value: Any, max_length: int = 500) -> str:
        """Format a state field for display."""
        if value is None:
            return "None"

        if isinstance(value, (dict, list)):
            formatted = json.dumps(value, indent=2)
            if len(formatted) > max_length:
                return formatted[:max_length] + f"\n... (truncated, {len(formatted)} chars total)"
            return formatted

        if isinstance(value, str):
            if len(value) > max_length:
                return value[:max_length] + f"\n... (truncated, {len(value)} chars total)"
            return value

        return str(value)

    def display_state_snapshot(self, node_name: str, state: Dict[str, Any], duration: float, call_number: int):
        """Display the complete state after a node executes."""
        self.step_count += 1

        print(f"\n{'='*80}")
        print(f"STEP {self.step_count}: After '{node_name}' Node (Call #{call_number})")
        print(f"⏱️  Execution Time: {duration:.3f} seconds")
        print(f"{'='*80}")

        # Core workflow fields
        print(f"\n📋 CORE WORKFLOW STATE:")
        print(f"  • original_query: {state.get('original_query', 'N/A')}")
        print(f"  • refined_query: {state.get('refined_query', 'N/A')}")
        print(f"  • iteration_count: {state.get('iteration_count', 0)}")
        print(f"  • max_iterations: {state.get('max_iterations', 3)}")

        # Intent metadata
        if state.get('intent_metadata'):
            print(f"\n🎯 INTENT METADATA:")
            intent = state['intent_metadata']
            print(f"  • primary_intent: {intent.get('primary_intent', 'N/A')}")
            print(f"  • key_entities: {intent.get('key_entities', [])}")
            print(f"  • required_facts_count: {intent.get('required_facts_count', 'N/A')}")
            print(f"  • detail_level: {intent.get('detail_level', 'N/A')}")
            print(f"  • suggested_keywords: {intent.get('suggested_keywords', [])}")

        # Search results
        if state.get('search_results'):
            print(f"\n🔍 SEARCH RESULTS ({len(state['search_results'])} results):")
            for i, result in enumerate(state['search_results'], 1):
                print(f"  [{i}] {result.get('title', 'No title')}")
                print(f"      URL: {result.get('link', 'N/A')}")
                snippet = result.get('snippet', 'No snippet')[:100]
                print(f"      Snippet: {snippet}...")

            # Show ChromaDB status if tools are enabled
            if self.enable_tools and node_name == 'fetch_sources':
                from vector_store_service import get_vector_store_service
                vector_service = get_vector_store_service()
                total_docs = vector_service.get_collection_count()

                # Get embedding model info
                embedding_model = vector_service.embeddings.model

                print(f"\n📦 CHROMADB STATUS:")
                print(f"  • Total documents in database: {total_docs}")
                print(f"  • Embedding model: {embedding_model}")
                print(f"  • Just cached: {len(state['search_results'])} new documents")
                print(f"  • Relevance threshold: {self.local_docs_score_threshold} (L2 distance, lower = more similar)")
                if self.local_docs_score_threshold == 0:
                    print(f"    → Filtering DISABLED (all results included)")
                else:
                    print(f"    → Only results with score ≤ {self.local_docs_score_threshold} are included")

                # Check if cached results were included
                if "CACHED RESULTS FROM PREVIOUS RESEARCH" in state.get('formatted_results', ''):
                    print(f"  • Retrieved cached results: Yes (integrated into analysis)")

                    # Show cached results with scores if available
                    vector_db_results = state.get('vector_db_results', [])
                    if vector_db_results:
                        print(f"\n💾 CACHED RESULTS ({len(vector_db_results)} documents passed filter):")
                        for i, result in enumerate(vector_db_results, 1):
                            title = result.get('title', 'Untitled')[:60]
                            score = result.get('relevance_score', 'N/A')
                            print(f"  [{i}] {title} (score: {score})")
                else:
                    print(f"  • Retrieved cached results: No (database may be empty or no relevant matches)")

                # Show detailed local document retrieval
                retrieved_docs = state.get('retrieved_documents', [])
                if retrieved_docs:
                    print(f"\n📄 RETRIEVED LOCAL DOCUMENTS ({len(retrieved_docs)} documents passed filter):")
                    for i, doc in enumerate(retrieved_docs, 1):
                        title = doc.get('title', 'Untitled')
                        score = doc.get('relevance_score', 'N/A')
                        print(f"  [{i}] {title} (score: {score})")
                        print(f"      Path: {doc.get('file_path', 'N/A')}")
                        content_preview = doc.get('content', '')[:100]
                        print(f"      Preview: {content_preview}...")
                else:
                    print(f"\n📄 RETRIEVED LOCAL DOCUMENTS: None")
                    if self.local_docs_score_threshold > 0:
                        print(f"    → No documents met relevance threshold of {self.local_docs_score_threshold}")

        # Evaluation result
        if state.get('evaluation_result'):
            print(f"\n📊 SEARCH EVALUATION:")
            eval_result = state['evaluation_result']
            print(f"  • recommendation: {eval_result.get('recommendation', 'N/A')}")
            print(f"  • relevance_score: {eval_result.get('relevance_score', 'N/A')}/10")
            print(f"  • coverage_score: {eval_result.get('coverage_score', 'N/A')}/10")
            print(f"  • quality_score: {eval_result.get('quality_score', 'N/A')}/10")
            if eval_result.get('identified_gaps'):
                print(f"  • identified_gaps: {eval_result['identified_gaps']}")

        # Ranked documents
        if state.get('ranked_documents') and state['ranked_documents'].get('ranked_documents'):
            print(f"\n📋 RANKED DOCUMENTS:")
            for i, doc in enumerate(state['ranked_documents']['ranked_documents'][:3], 1):
                print(f"  [{i}] {doc.get('title', 'No title')}")
                print(f"      Relevance: {doc.get('relevance_score', 'N/A')}/10")
                print(f"      Rationale: {doc.get('relevance_rationale', 'N/A')[:100]}...")

        # Validated facts
        if state.get('validated_facts') and state['validated_facts'].get('validated_facts'):
            print(f"\n🧠 VALIDATED FACTS ({len(state['validated_facts']['validated_facts'])} facts):")
            for i, fact in enumerate(state['validated_facts']['validated_facts'], 1):
                print(f"  [{i}] {fact.get('fact', 'N/A')}")
                print(f"      Confidence: {fact.get('confidence', 'N/A')}")

                # Show which sources support this fact (KEY for verifying local doc usage!)
                supporting = fact.get('supporting_sources', [])
                print(f"      Supporting sources: {supporting}")

                # Highlight if local docs were used
                local_doc_refs = [s for s in supporting if 'Local Document' in str(s) or 'local' in str(s).lower()]
                if local_doc_refs:
                    print(f"      🎯 Uses LOCAL DOCUMENTS: {local_doc_refs}")

                if fact.get('evidence_summary'):
                    summary = fact.get('evidence_summary', '')[:150]
                    print(f"      Evidence: {summary}...")

        # Summary
        if state.get('summary'):
            print(f"\n📝 SUMMARY (COMPLETE):")
            summary = state['summary']
            print(f"  {summary}")

        # Final output
        if state.get('output'):
            print(f"\n🎨 FINAL OUTPUT (COMPLETE):")
            output = state['output']
            print(f"  {output}")

        print(f"\n{'='*80}\n")

    def display_routing_decision(self, from_node: str, to_node: str, reason: str = ""):
        """Display routing decisions between nodes."""
        print(f"\n🔀 ROUTING DECISION:")
        print(f"   From: '{from_node}' → To: '{to_node}'")
        if reason:
            print(f"   Reason: {reason}")
        print()

    def display_timing_statistics(self):
        """Display comprehensive timing statistics at the end."""
        total_time = self.workflow_end_time - self.workflow_start_time

        print(f"\n{'='*80}")
        print(f"⏱️  TIMING STATISTICS")
        print(f"{'='*80}\n")

        print(f"📊 TOTAL WORKFLOW TIME: {total_time:.3f} seconds ({total_time/60:.2f} minutes)\n")

        # Group timings by node
        node_timings_grouped = {}
        for node_name, duration, call_number in self.node_timings:
            if node_name not in node_timings_grouped:
                node_timings_grouped[node_name] = []
            node_timings_grouped[node_name].append((call_number, duration))

        print(f"📋 NODE EXECUTION TIMES (in execution order):\n")

        # Get node names in execution order (order of first appearance)
        node_order = []
        for node_name, _, _ in self.node_timings:
            if node_name not in node_order:
                node_order.append(node_name)

        # Display each node's timing(s) in execution order
        for node_name in node_order:
            timings = node_timings_grouped[node_name]
            total_node_time = sum(duration for _, duration in timings)
            avg_time = total_node_time / len(timings)
            percentage = (total_node_time / total_time) * 100

            if len(timings) == 1:
                print(f"  • {node_name}:")
                print(f"      Time: {timings[0][1]:.3f}s ({percentage:.1f}% of total)")
            else:
                print(f"  • {node_name} (called {len(timings)} times):")
                for call_num, duration in timings:
                    print(f"      Call #{call_num}: {duration:.3f}s")
                print(f"      Total: {total_node_time:.3f}s | Avg: {avg_time:.3f}s ({percentage:.1f}% of total)")

        print(f"\n{'='*80}\n")

        # Display execution order with cumulative time
        print(f"📈 EXECUTION TIMELINE (Cumulative):\n")
        cumulative = 0
        for i, (node_name, duration, call_number) in enumerate(self.node_timings, 1):
            cumulative += duration
            print(f"  {i}. {node_name} (Call #{call_number}): +{duration:.3f}s → {cumulative:.3f}s total")

        print(f"\n{'='*80}\n")

    def run_transparent_workflow(self, query: str):
        """Run the workflow with complete transparency."""
        print(f"\n{'#'*80}")
        print(f"# TRANSPARENT WORKFLOW EXECUTION")
        print(f"# Query: {query}")
        if self.skip_formatting:
            print(f"# Mode: FAST (skip formatting)")
        else:
            print(f"# Mode: FULL (with formatting)")
        if self.enable_tools:
            print(f"# Tools: ENABLED (caching to ChromaDB)")
            print(f"#   - Cached results: {self.cached_results_count} max")
            print(f"#   - Local docs: {self.local_docs_count} max")
            print(f"#   - Relevance threshold: {self.local_docs_score_threshold}")
        else:
            print(f"# Tools: DISABLED")
        if self.detailed_output:
            print(f"# Detailed output: ENABLED")
        print(f"{'#'*80}\n")

        # Initialize state
        initial_state: ResearchState = {
            "original_query": query,
            "refined_query": "",
            "search_results": [],
            "formatted_results": "",
            "summary": "",
            "output": "",
            "intent_metadata": {},
            "evaluation_result": {},
            "validated_facts": {},
            "ranked_documents": {},
            "iteration_count": 0,
            "max_iterations": 3,
            "vector_db_results": None,
            "retrieved_documents": None,
            "tool_outputs": None,
            "metrics": {},
            "evaluation_metadata": {}
        }

        print(f"📥 INITIAL STATE:")
        print(f"  • Query: {query}")
        print(f"  • Max iterations: {initial_state['max_iterations']}")

        # Use stream to get intermediate states
        print(f"\n🚀 STARTING WORKFLOW EXECUTION...\n")

        previous_node = "START"
        self.workflow_start_time = time.time()
        node_start_time = self.workflow_start_time

        try:
            for step_output in self.workflow.graph.stream(initial_state):
                # step_output is a dict with node_name: state
                for node_name, state in step_output.items():
                    if node_name != "__end__":
                        # Calculate time for this node
                        node_end_time = time.time()
                        duration = node_end_time - node_start_time

                        # Track call count for this node
                        if node_name not in self.node_call_counts:
                            self.node_call_counts[node_name] = 0
                        self.node_call_counts[node_name] += 1
                        call_number = self.node_call_counts[node_name]

                        # Record timing
                        self.node_timings.append((node_name, duration, call_number))

                        # Display routing if not the first node
                        if previous_node != "START":
                            self.display_routing_decision(previous_node, node_name)

                        # Display state after this node
                        self.display_state_snapshot(node_name, state, duration, call_number)
                        previous_node = node_name

                        # Reset timer for next node
                        node_start_time = time.time()

            self.workflow_end_time = time.time()

            print(f"\n{'#'*80}")
            print(f"# WORKFLOW EXECUTION COMPLETE")
            print(f"# Total steps executed: {self.step_count}")
            print(f"{'#'*80}\n")

            # Display timing statistics
            self.display_timing_statistics()

        except Exception as e:
            print(f"\n❌ ERROR during workflow execution:")
            print(f"   {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point for transparency test."""
    # Check for flags
    skip_formatting = False
    enable_tools = False
    detailed_output = False
    cached_results_count = 3
    local_docs_count = 2
    local_docs_score_threshold = 0.5

    args = sys.argv[1:]

    if "--skip-formatting" in args:
        skip_formatting = True
        args.remove("--skip-formatting")

    if "--enable-tools" in args:
        enable_tools = True
        args.remove("--enable-tools")

    if "--detailed-output" in args:
        detailed_output = True
        args.remove("--detailed-output")

    # Parse --relevance-threshold=<value>
    threshold_args = [arg for arg in args if arg.startswith("--relevance-threshold=")]
    if threshold_args:
        try:
            local_docs_score_threshold = float(threshold_args[0].split("=")[1])
            args.remove(threshold_args[0])
            if local_docs_score_threshold < 0:
                print("❌ Error: Relevance threshold must be >= 0")
                print("   Typical values: 0 (no filtering), 0.3 (strict), 0.5 (moderate), 2.0 (lenient)")
                sys.exit(1)
        except ValueError:
            print("❌ Error: Invalid relevance threshold value. Must be a number (e.g., 0.3)")
            sys.exit(1)

    # Parse --cached-results=<count>
    cached_args = [arg for arg in args if arg.startswith("--cached-results=")]
    if cached_args:
        try:
            cached_results_count = int(cached_args[0].split("=")[1])
            args.remove(cached_args[0])
            if cached_results_count < 0:
                print("❌ Error: Cached results count must be >= 0")
                sys.exit(1)
        except ValueError:
            print("❌ Error: Invalid cached results count. Must be an integer")
            sys.exit(1)

    # Parse --local-docs=<count>
    local_docs_args = [arg for arg in args if arg.startswith("--local-docs=")]
    if local_docs_args:
        try:
            local_docs_count = int(local_docs_args[0].split("=")[1])
            args.remove(local_docs_args[0])
            if local_docs_count < 0:
                print("❌ Error: Local docs count must be >= 0")
                sys.exit(1)
        except ValueError:
            print("❌ Error: Invalid local docs count. Must be an integer")
            sys.exit(1)

    if len(args) < 1:
        print("Usage: python test_workflow_transparency.py [OPTIONS] \"your query here\"")
        print("\nOptions:")
        print("  --skip-formatting              Skip the format_output node to speed up execution")
        print("  --enable-tools                 Enable tools (vector search, document loader, ChromaDB caching)")
        print("  --detailed-output              Include detailed fact validation in output")
        print("  --relevance-threshold=<value>  Set relevance threshold (default: 0.5)")
        print("                                 0 = no filtering, 0.3 = strict, 0.5 = moderate, 2.0 = lenient")
        print("  --cached-results=<count>       Max cached results to retrieve (default: 3)")
        print("  --local-docs=<count>           Max local documents to retrieve (default: 2)")
        print("\nExamples:")
        print('  python test_workflow_transparency.py "Find 3 facts about Python programming"')
        print('  python test_workflow_transparency.py --skip-formatting "Find 3 facts about Python"')
        print('  python test_workflow_transparency.py --enable-tools "Find 3 facts about Python"')
        print('  python test_workflow_transparency.py --enable-tools --relevance-threshold=0.3 "Query with strict filtering"')
        print('  python test_workflow_transparency.py --enable-tools --cached-results=5 --local-docs=3 "More results"')
        sys.exit(1)

    query = args[0]

    debugger = WorkflowDebugger(
        skip_formatting=skip_formatting,
        enable_tools=enable_tools,
        cached_results_count=cached_results_count,
        local_docs_count=local_docs_count,
        local_docs_score_threshold=local_docs_score_threshold,
        detailed_output=detailed_output
    )
    debugger.run_transparent_workflow(query)


if __name__ == "__main__":
    main()
