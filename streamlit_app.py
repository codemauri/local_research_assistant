"""
Streamlit Web UI for Local Research Assistant
==============================================

A chat-based web interface for conducting AI-powered research with:
- Interactive chat history
- Configurable search parameters
- Real-time progress updates
- Source transparency
- Local document integration

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import time
from datetime import datetime
from workflow import ResearchWorkflow
from vector_store_service import get_vector_store_service


# Page configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better chat UI
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stChatMessage[data-testid="chat-message-assistant"] {
        background-color: #f0f2f6;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    .source-item {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #007bff;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "workflow_config" not in st.session_state:
    st.session_state.workflow_config = {
        "enable_tools": True,
        "detailed_output": False,
        "relevance_threshold": 0.5,
        "cached_results_count": 3,
        "local_docs_count": 2
    }


def format_research_output(output: str, detailed: bool = False) -> None:
    """Format and display research results in a structured way."""

    # Split output into sections
    sections = output.split("\n## ")

    for i, section in enumerate(sections):
        if i == 0 and not section.startswith("# "):
            # First section without header
            st.markdown(section)
            continue

        # Parse section header and content
        lines = section.split("\n", 1)
        header = lines[0].replace("# ", "").strip()
        content = lines[1] if len(lines) > 1 else ""

        # Display based on section type
        if "Sources" in header:
            with st.expander("📚 **Sources**", expanded=True):
                st.markdown(content)
        elif "Detailed Fact Validation" in header:
            with st.expander("🔍 **Detailed Fact Validation**", expanded=False):
                st.markdown(content)
        elif "Summary" in header or "Facts" in header:
            st.markdown(f"### {header}")
            st.markdown(content)
        else:
            st.markdown(f"### {header}")
            st.markdown(content)


def get_database_stats() -> dict:
    """Get ChromaDB statistics."""
    try:
        vector_service = get_vector_store_service()
        total_docs = vector_service.get_collection_count()
        embedding_model = vector_service.embeddings.model

        return {
            "total_docs": total_docs,
            "embedding_model": embedding_model,
            "status": "connected"
        }
    except Exception as e:
        return {
            "total_docs": 0,
            "embedding_model": "unknown",
            "status": f"error: {str(e)}"
        }


# Sidebar - Configuration
with st.sidebar:
    st.title("⚙️ Configuration")

    # Tools settings
    st.subheader("🔧 Tools & Search")

    enable_tools = st.toggle(
        "Enable Vector Search & Caching",
        value=st.session_state.workflow_config["enable_tools"],
        help="Enable ChromaDB for caching results and searching local documents"
    )
    st.session_state.workflow_config["enable_tools"] = enable_tools

    if enable_tools:
        st.divider()

        # ChromaDB stats
        db_stats = get_database_stats()
        st.metric("Database Documents", db_stats["total_docs"])
        st.caption(f"Embedding: `{db_stats['embedding_model']}`")

        st.divider()

        # Search parameters
        relevance_threshold = st.slider(
            "Relevance Threshold",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.workflow_config["relevance_threshold"],
            step=0.1,
            help="L2 distance threshold (0 = no filter, 0.3 = strict, 0.5 = moderate, 2.0 = lenient)"
        )
        st.session_state.workflow_config["relevance_threshold"] = relevance_threshold

        if relevance_threshold == 0:
            st.caption("⚠️ No filtering - all results included")
        elif relevance_threshold <= 0.3:
            st.caption("🎯 Strict filtering - only highly relevant")
        elif relevance_threshold <= 0.7:
            st.caption("📊 Moderate filtering - balanced")
        else:
            st.caption("🌐 Lenient filtering - broader results")

        col1, col2 = st.columns(2)
        with col1:
            cached_results = st.number_input(
                "Cached Results",
                min_value=0,
                max_value=10,
                value=st.session_state.workflow_config["cached_results_count"],
                help="Number of cached web results to retrieve"
            )
            st.session_state.workflow_config["cached_results_count"] = cached_results

        with col2:
            local_docs = st.number_input(
                "Local Docs",
                min_value=0,
                max_value=10,
                value=st.session_state.workflow_config["local_docs_count"],
                help="Number of local documents to retrieve"
            )
            st.session_state.workflow_config["local_docs_count"] = local_docs

    st.divider()

    # Output settings
    st.subheader("📄 Output Settings")

    detailed_output = st.toggle(
        "Detailed Output",
        value=st.session_state.workflow_config["detailed_output"],
        help="Include detailed fact validation with sources and confidence scores"
    )
    st.session_state.workflow_config["detailed_output"] = detailed_output

    st.divider()

    # Actions
    st.subheader("🗑️ Actions")

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Info section
    st.divider()
    st.subheader("ℹ️ About")
    st.caption("""
    **AI Research Assistant**

    An intelligent research assistant that:
    - Conducts web searches
    - Caches results for reuse
    - Searches local documents
    - Validates facts across sources
    - Provides transparent citations

    Built with LangGraph + Ollama + ChromaDB
    """)


# Main content area
st.title("🔍 AI Research Assistant")
st.caption("Ask me anything - I'll research and provide sourced answers")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            if "formatted" in message and message["formatted"]:
                # Display formatted research output
                format_research_output(message["content"], detailed=message.get("detailed", False))
            else:
                # Display plain message
                st.markdown(message["content"])
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Enter your research question..."):
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response with progress
    with st.chat_message("assistant"):
        # Progress placeholder
        status_placeholder = st.empty()
        result_placeholder = st.empty()

        try:
            # Show initial status
            with status_placeholder.status("🔍 Researching...", expanded=True) as status:
                st.write("📋 Parsing your question...")
                time.sleep(0.5)

                st.write("🔄 Refining search query...")
                time.sleep(0.5)

                st.write("🌐 Fetching sources...")

                config = st.session_state.workflow_config

                # Create and run workflow
                workflow = ResearchWorkflow(
                    skip_formatting=False,
                    enable_tools=config["enable_tools"],
                    cached_results_count=config["cached_results_count"],
                    local_docs_count=config["local_docs_count"],
                    local_docs_score_threshold=config["relevance_threshold"],
                    detailed_output=config["detailed_output"]
                )

                st.write("🧠 Analyzing and synthesizing information...")

                # Run workflow
                output = workflow.run(prompt)

                st.write("✅ Research complete!")
                status.update(label="✅ Research Complete!", state="complete", expanded=False)

            # Clear status and display results
            status_placeholder.empty()

            # Display formatted output
            with result_placeholder.container():
                format_research_output(output, detailed=config["detailed_output"])

            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": output,
                "timestamp": datetime.now().isoformat(),
                "formatted": True,
                "detailed": config["detailed_output"]
            })

        except Exception as e:
            status_placeholder.empty()

            # Display error
            st.error(f"❌ Error during research: {str(e)}")

            # Show detailed error in expander
            with st.expander("🔍 Error Details"):
                st.code(str(e))
                import traceback
                st.code(traceback.format_exc())

            # Add error message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ I encountered an error: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "formatted": False
            })


# Footer
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"💬 Messages: {len(st.session_state.messages)}")

with col2:
    if st.session_state.workflow_config["enable_tools"]:
        st.caption("🔧 Tools: **Enabled**")
    else:
        st.caption("🔧 Tools: Disabled")

with col3:
    if st.session_state.workflow_config["detailed_output"]:
        st.caption("📄 Output: **Detailed**")
    else:
        st.caption("📄 Output: Standard")
