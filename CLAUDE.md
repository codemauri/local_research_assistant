# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research assistant system built with LangGraph that orchestrates a multi-step agentic workflow for web research. The system takes natural language queries, refines them, searches the web using Google Custom Search API, and generates structured summaries using a local Ollama LLM.

## Environment Setup

### Prerequisites
- Ollama must be running locally (default: http://localhost:11434)
- A model installed in Ollama (default: llama3.2)
- Google Custom Search API credentials (API key and CSE ID)

### Setup Commands

```bash
# Using Conda (recommended)
conda env create -f environment.yml
conda activate research_assistant

# Or using pip
pip install -r requirements.txt

# Ensure Ollama is running
ollama serve

# Pull the model if needed
ollama pull llama3.2
```

### Environment Variables
Create a `.env` file (gitignored) with:
```
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_custom_search_engine_id_here
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

## Running the Application

```bash
# Interactive mode
python main.py

# Command line mode
python main.py "Find 3 facts about AI safety"
```

## Architecture

### LangGraph Workflow
The system uses a directed graph with four sequential nodes:

1. **refine_query** - Uses Ollama LLM to convert natural language requests into optimized search queries
2. **fetch_sources** - Calls Google Custom Search API to retrieve web results (default: 5 results, max: 10 per API limitation)
3. **summarize** - Uses Ollama LLM to extract key facts from search results
4. **format_output** - Structures results with citations and metadata

The workflow is defined in `workflow.py` using LangGraph's `StateGraph` with typed state (`ResearchState`) that flows through all nodes.

### State Management
The `ResearchState` TypedDict tracks:
- `original_query`: User's input
- `refined_query`: LLM-optimized search query
- `search_results`: List of Google search results
- `formatted_results`: Search results formatted for LLM consumption
- `summary`: LLM-generated summary
- `output`: Final formatted output with citations

### Service Layer
- **OllamaLLMService** (`llm_service.py`): Wraps LangChain's Ollama integration with two prompting methods:
  - `refine_query()`: Query optimization
  - `summarize_search_results()`: Fact extraction (supports variable number of facts via regex parsing)

- **GoogleSearchService** (`search_service.py`): Google Custom Search API wrapper
  - Validates credentials on initialization
  - Returns structured results (title, link, snippet)
  - Formats results for LLM processing

### Configuration
`config.py` loads environment variables via python-dotenv. All services pull configuration from this central module.

## Key Implementation Details

### LangChain Version Compatibility
`llm_service.py` includes a fallback import pattern for different LangChain versions:
```python
try:
    from langchain_ollama import OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama as OllamaLLM
```

### Dynamic Fact Extraction
The workflow extracts the number of facts from the query using regex (`_extract_num_facts()` in `workflow.py`), defaulting to 3 if not specified.

### Error Handling
- Google API credentials are validated on service initialization
- LLM errors are caught and re-raised with context
- Main entry point catches and displays configuration errors vs. runtime errors separately

## Customization Points

### Changing Number of Search Results
Modify `workflow.py:57`:
```python
results = self.search_service.search(state['refined_query'], num_results=10)
```

### Changing LLM Model
Update `.env`:
```
OLLAMA_MODEL=your_preferred_model
```

### Modifying Output Format
Edit `_format_output_node()` in `workflow.py` (lines 77-97)

### Adjusting Prompts
- Query refinement prompt: `llm_service.py:68-72`
- Summarization prompt: `llm_service.py:39-50`

## Custom Slash Commands

This project includes custom slash commands in `.claude/commands/` for common workflows:

### Development Workflow Commands
- **/doit** - Read `prompt.md` from project root and execute the task described in it
- **/udc** - Update documentation and commit changes

### Idea Management Commands
- **/idea-create [feedback]** - Create or append to `ideas.md` with enhancement ideas. Takes optional user feedback into account
- **/idea-next [feedback]** - Read `ideas.md`, select next idea to work on (considering optional feedback), create feature branch, and implement it. Uses `@research-documentation-agent` if research is needed
- **/idea-done [feedback]** - Mark current idea as completed in `ideas.md`, move to completed section, commit changes, and merge feature branch to main

### GitHub Integration Commands
- **/create-issue [description]** - Create a new GitHub issue using `gh` CLI with auto-generated title summary
- **/fix-issue #[number]** - Complete workflow to fix an issue:
  1. Checkout feature branch
  2. Understand the issue
  3. Locate and fix code
  4. Add tests (integration tests for iOS, unit tests for Rust)
  5. Update `make test` target
  6. Run `make test` to verify
  7. Prepare PR description
- **/merge-pr** - Create PR for current feature branch (if none exists) and merge it

**Note**: The `/fix-issue` command mentions schema-first design pattern, schemas directory, and rustls preference - these may not apply to this project but are part of the inherited command template.

## Custom Agents

### @research-documentation-agent
Located in `.claude/agents/research-documentation-agent.md`

**Purpose**: Specialized agent for conducting technical research, gathering documentation from web sources, and creating comprehensive markdown documentation files.

**When to Use Proactively**:
- User is exploring a new technology or library
- User mentions unfamiliar technical concepts or APIs
- User is debugging or troubleshooting complex issues
- Beginning a new project or feature that requires technical knowledge

**Key Capabilities**:
- Examines existing `docs/research/` folder to avoid duplicate research
- Conducts comprehensive web searches using Brave Search
- Scrapes and extracts content from authoritative sources
- Creates well-structured markdown documentation following style guidelines
- Validates information across multiple sources
- Preserves proper attribution and source URLs

**Documentation Standards**:
- Saves files to `docs/research/` with naming format: `topic-name-YYYY-MM-DD.md`
- Includes metadata: research date, last updated, sources
- Follows comprehensive template with sections: Overview, Key Concepts, Technical Details, Best Practices, Common Pitfalls, Performance Considerations, Examples, Further Reading
- Maintains research index at `docs/research/INDEX.md`

**Usage Example**:
```
"I need to implement signed distance functions for ray marching"
→ Launch @research-documentation-agent to gather comprehensive SDF and ray marching documentation
```

## Documentation Style Guide

This project follows strict documentation standards defined in `.claude/DOCUMENTATION_STYLE_GUIDE.md`:

### Key Requirements
- **Never use ASCII art** - Always use Mermaid diagrams
- **Always include TOC** for documents > 500 words
- **Use dark backgrounds** with white text in diagrams for high contrast
- **Specify language** in all code blocks
- **Test all examples** before documenting

### Color Scheme for Mermaid Diagrams
All diagrams must use high-contrast colors with white text (`color:#ffffff`):

| Component Type | Fill Color | Stroke Color | Usage |
|---------------|------------|--------------|-------|
| Primary/Main | `#e65100` | `#ff9800` | Main components, orchestrators |
| Active/Healthy | `#1b5e20` | `#4caf50` | Active services, healthy states |
| Failed/Unhealthy | `#b71c1c` | `#f44336` | Failed components, errors |
| Data Storage | `#0d47a1` | `#2196f3` | Redis, cache layers |
| Database | `#1a237e` | `#3f51b5` | PostgreSQL, persistent storage |
| Client/External | `#4a148c` | `#9c27b0` | CLI, TUI, API clients |
| Warning | `#ff6f00` | `#ffa726` | Decision points, warnings |
| Neutral/Info | `#37474f` | `#78909c` | Stopped services, info |

### Document Structure Template
Every documentation file should include:
1. **Title (H1)** - Clear, descriptive
2. **Brief Description** - 1-2 sentences after title
3. **Table of Contents** - For docs > 500 words
4. **Overview Section** - Context and scope
5. **Main Content** - Logical sections with proper hierarchy
6. **Related Documentation** - Links to relevant resources

### Writing Style
- **Professional but approachable** - Technical without unnecessary jargon
- **Direct and actionable** - Focus on what reader needs to do
- **Present tense** - "The system uses..." not "will use..."
- **Active voice** - "The API returns data" not "Data is returned"
- Use callout boxes with emoji indicators: 📝 Note, ⚠️ Warning, ✅ Tip, 🚫 Deprecated, 🔒 Security

For complete style guide details, see `.claude/DOCUMENTATION_STYLE_GUIDE.md`
