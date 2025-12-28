"""
LangChain Tools for Research Assistant
========================================

Custom tools that extend the agent's capabilities using the @tool decorator.
Tools can access workflow state via ToolRuntime and are called dynamically by the LLM.

Design Pattern:
- Use @tool decorator for LangChain integration
- Type hints are required (used for schema generation)
- ToolRuntime provides access to workflow state
- Tools return simple strings that LLM can understand
"""

from typing import Optional
from langchain_core.tools import tool
from vector_store_service import get_vector_store_service
import os


@tool
def search_cached_research(query: str, top_k: int = 3) -> str:
    """
    Search previously cached research results for semantically similar information.

    Useful when the current query might be related to previous research or when
    you need to find connections between different topics.

    Args:
        query: The search query to find similar content
        top_k: Number of similar results to return (default: 3)

    Returns:
        String containing similar results with titles and snippets
    """
    vector_service = get_vector_store_service()
    results = vector_service.similarity_search(query, top_k=top_k)

    if not results:
        return "No similar cached research found."

    # Format results for LLM
    formatted = f"Found {len(results)} similar cached results:\n\n"
    for i, result in enumerate(results, 1):
        formatted += f"{i}. {result.get('title', 'Untitled')}\n"
        formatted += f"   {result.get('content', '')[:200]}...\n"
        if result.get('link'):
            formatted += f"   Source: {result['link']}\n"
        formatted += "\n"

    return formatted


@tool
def load_local_document(file_path: str) -> str:
    """
    Load and retrieve content from a local document file.

    Supports text files (.txt) and markdown files (.md).
    Useful for incorporating local knowledge or documentation into research.

    Args:
        file_path: Path to the document file (relative to ./documents/ directory)

    Returns:
        String containing the document content, or error message if file not found
    """
    # Security: Only allow files from documents directory
    base_dir = "./documents"
    safe_path = os.path.join(base_dir, file_path)

    # Prevent directory traversal
    if not os.path.abspath(safe_path).startswith(os.path.abspath(base_dir)):
        return f"Error: Access denied. Files must be in {base_dir} directory."

    # Check if file exists
    if not os.path.exists(safe_path):
        return f"Error: File not found: {file_path}"

    # Check file extension
    allowed_extensions = {'.txt', '.md'}
    _, ext = os.path.splitext(safe_path)
    if ext.lower() not in allowed_extensions:
        return f"Error: Unsupported file type. Allowed: {', '.join(allowed_extensions)}"

    # Read file
    try:
        with open(safe_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Limit content length to avoid overwhelming the LLM
        max_length = 2000
        if len(content) > max_length:
            content = content[:max_length] + f"\n\n[Content truncated - {len(content)} total characters]"

        return f"Content from {file_path}:\n\n{content}"

    except Exception as e:
        return f"Error reading file: {str(e)}"


# List of all available tools
AVAILABLE_TOOLS = [
    search_cached_research,
    load_local_document,
]


def get_tools():
    """Get all available tools for the agent."""
    return AVAILABLE_TOOLS
