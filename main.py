"""Main entry point for the research assistant."""
import sys
from workflow import ResearchWorkflow


def main():
    """Run the research assistant."""
    # Check for flags
    skip_formatting = False
    enable_tools = False
    detailed_output = False
    relevance_threshold = 0.5  # Default threshold
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
            relevance_threshold = float(threshold_args[0].split("=")[1])
            args.remove(threshold_args[0])
            if relevance_threshold < 0:
                print("❌ Error: Relevance threshold must be >= 0")
                print("   Typical values: 0 (no filtering), 0.3 (strict), 0.5 (moderate), 2.0 (lenient)")
                return
        except ValueError:
            print("❌ Error: Invalid relevance threshold value. Must be a number (e.g., 0.3)")
            return

    if len(args) > 0:
        # Command line mode
        query = " ".join(args)
    else:
        # Interactive mode
        print("🤖 Research Assistant with LangGraph Workflow")
        print("=" * 50)
        query = input("\nEnter your research query: ").strip()

    if not query:
        print("❌ Error: Please provide a research query")
        print("\nUsage: python main.py [--skip-formatting] [--enable-tools] [--detailed-output] [--relevance-threshold=<value>] \"your query here\"")
        print("       Threshold values: 0 (no filtering), 0.3 (strict), 0.5 (moderate, default), 2.0 (lenient)")
        return

    try:
        print(f"\n🚀 Processing: {query}")
        if skip_formatting:
            print("⚡ Skip formatting enabled - using raw summary as output")
        if enable_tools:
            print("🔧 Tools enabled - caching results & enabling semantic search")
            if relevance_threshold != 0.5:
                print(f"🎯 Relevance threshold: {relevance_threshold} (lower = stricter filtering)")
        if detailed_output:
            print("📊 Detailed output enabled - includes validated facts with sources")
        if not skip_formatting and not enable_tools and not detailed_output:
            print()

        workflow = ResearchWorkflow(
            skip_formatting=skip_formatting,
            enable_tools=enable_tools,
            local_docs_score_threshold=relevance_threshold,
            detailed_output=detailed_output
        )
        result = workflow.run(query)
        
        print("\n" + "=" * 50)
        print("📊 RESEARCH RESULTS")
        print("=" * 50 + "\n")
        print(result)
        print("\n" + "=" * 50)
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\nPlease ensure you have set up your .env file with:")
        print("- GOOGLE_API_KEY")
        print("- GOOGLE_CSE_ID")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

