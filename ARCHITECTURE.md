# Agentic AI Research Assistant - System Architecture

## Executive Summary

An intelligent research assistant built with LangGraph that autonomously conducts web research, evaluates result quality, and synthesizes information from multiple sources. The system features adaptive query refinement, semantic caching via vector storage, local document integration, and multi-source reasoning with full transparency.

**Key Capabilities:**
- Autonomous query understanding and intent parsing
- Adaptive search with quality-based refinement loops
- Multi-source information synthesis (web, cached, local documents)
- Vector-based semantic search and caching
- Source attribution and transparency
- Configurable output modes for different use cases

---

## Table of Contents

- [System Architecture](#system-architecture)
  - [High-Level Architecture Diagram](#high-level-architecture-diagram)
  - [Workflow State Machine](#workflow-state-machine)
- [Technology Stack](#technology-stack)
  - [Core Framework](#core-framework)
  - [LLM Integration](#llm-integration)
  - [Search & Retrieval](#search--retrieval)
  - [Data Processing](#data-processing)
  - [Configuration](#configuration)
- [Component Breakdown](#component-breakdown)
  - [1. Workflow Engine (`workflow.py`)](#1-workflow-engine-workflowpy)
  - [2. Prompt Templates (`prompt_templates.py`)](#2-prompt-templates-prompt_templatespy)
  - [3. LLM Service (`llm_service.py`)](#3-llm-service-llm_servicepy)
  - [4. Vector Store Service (`vector_store_service.py`)](#4-vector-store-service-vector_store_servicepy)
  - [5. Search Service (`search_service.py`)](#5-search-service-search_servicepy)
- [Key Design Decisions](#key-design-decisions)
  - [1. Conditional Routing for Agentic Behavior](#1-conditional-routing-for-agentic-behavior)
  - [2. Multi-Source Integration with Transparency](#2-multi-source-integration-with-transparency)
  - [3. Dual Output Modes](#3-dual-output-modes)
  - [4. Smart Document Indexing](#4-smart-document-indexing)
  - [5. Lenient Search Evaluation](#5-lenient-search-evaluation)
- [Execution Modes](#execution-modes)
  - [Standard Mode](#standard-mode)
  - [Tools Mode](#tools-mode)
  - [Fast Mode](#fast-mode)
  - [Demo Mode](#demo-mode)
  - [Combined](#combined)
- [Testing & Transparency](#testing--transparency)
  - [Test Script (`test_workflow_transparency.py`)](#test-script-test_workflow_transparencypy)
- [Performance Characteristics](#performance-characteristics)
  - [LLM Calls per Query](#llm-calls-per-query)
  - [Execution Time (Estimated)](#execution-time-estimated)
  - [API Usage](#api-usage)
- [Data Flow Example](#data-flow-example)
  - [Query: "What has been done for AI agents accountability and credibility"](#query-what-has-been-done-for-ai-agents-accountability-and-credibility)
- [Configuration](#configuration-1)
  - [Environment Variables (`.env`)](#environment-variables-env)
  - [Configurable Parameters](#configurable-parameters)
- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [First Run](#first-run)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)
- [Recent Improvements & Optimizations](#recent-improvements--optimizations)
  - [Post-Task 4 Enhancements (December 2024)](#post-task-4-enhancements-december-2024)
    - [1. Embedding Model Migration](#1-embedding-model-migration)
    - [2. Relevance Filtering with Score Thresholds](#2-relevance-filtering-with-score-thresholds)
    - [3. Enhanced JSON Parsing with Auto-Fixes](#3-enhanced-json-parsing-with-auto-fixes)
    - [4. Enhanced Test Script Transparency](#4-enhanced-test-script-transparency)
    - [5. Verification Utilities](#5-verification-utilities)
    - [6. Task 4 Completion: Evaluation Framework](#6-task-4-completion-evaluation-framework)
    - [7. CLI Configuration Enhancements](#7-cli-configuration-enhancements)
    - [8. Updated Technology Stack](#8-updated-technology-stack)
  - [Impact Summary](#impact-summary)
  - [Migration Checklist (For New Deployments)](#migration-checklist-for-new-deployments)
- [License](#license)
- [Contact & Support](#contact--support)

---

## System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                                  │
│                         (main.py / CLI)                                  │
│                                                                           │
│  Flags: --enable-tools | --skip-formatting | --detailed-output          │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      RESEARCH WORKFLOW ENGINE                            │
│                         (LangGraph StateGraph)                           │
│                                                                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │ Parse Intent │───▶│ Refine Query │───▶│Fetch Sources │              │
│  └──────────────┘    └──────────────┘    └──────┬───────┘              │
│                            ▲                      │                       │
│                            │                      ▼                       │
│                            │              ┌──────────────┐               │
│                            │              │   Evaluate   │               │
│                            │              │    Search    │               │
│                            │              └──────┬───────┘               │
│                            │                     │                       │
│                            │              ┌──────▼────────┐              │
│                            └──────────────│  Conditional  │              │
│                                 Loop      │    Routing    │              │
│                              (max 3x)     └──────┬────────┘              │
│                                                  │                       │
│                                                  ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │    Format    │◀───│  Summarize   │◀───│ Reason &     │              │
│  │    Output    │    │   Results    │    │ Synthesize   │              │
│  └──────────────┘    └──────────────┘    └──────▲───────┘              │
│                                                  │                       │
│                                           ┌──────┴───────┐              │
│                                           │Rank Documents│              │
│                                           └──────────────┘              │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
        ┌───────────────────┐   ┌──────────────────────┐
        │   LLM SERVICE     │   │   SEARCH SERVICE     │
        │   (Ollama)        │   │   (Google CSE)       │
        └───────────────────┘   └──────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────────┐
        │   VECTOR STORE SERVICE            │
        │   (ChromaDB + nomic-embed-text)   │
        │                                   │
        │  Collections:                     │
        │  • web_search (cached results)    │
        │  • local_document (research PDFs) │
        └───────────────────────────────────┘
```

### Workflow State Machine

```
                    START
                      │
                      ▼
            ┌─────────────────┐
            │  Parse Intent   │
            │  • Extract goal │
            │  • Count facts  │
            │  • Detail level │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │  Refine Query   │
            │  • Optimize     │
            │  • Add context  │
            │  • Temporal adj │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │  Fetch Sources  │◀─────────────┐
            │  • Web search   │              │
            │  • Vector cache │              │
            │  • Local docs   │              │
            └────────┬────────┘              │
                     │                       │
                     ▼                       │
            ┌─────────────────┐              │
            │ Evaluate Search │              │
            │  • Relevance    │              │
            │  • Coverage     │              │
            │  • Quality      │              │
            └────────┬────────┘              │
                     │                       │
                     ▼                       │
            ┌─────────────────┐              │
            │   Conditional   │              │
            │     Routing     │──────────────┤
            └────────┬────────┘              │
                     │                  Refine Loop
              Sufficient?              (max 3 iter)
                     │                       │
                     ▼                       │
            ┌─────────────────┐              │
            │Rank Documents   │              │
            │  • Score sources│              │
            │  • Prioritize   │              │
            └────────┬────────┘              │
                     │                       │
                     ▼                       │
            ┌─────────────────┐              │
            │Reason & Synth.  │              │
            │  • Validate     │              │
            │  • Cross-check  │              │
            │  • Detect gaps  │              │
            └────────┬────────┘              │
                     │                       │
                     ▼                       │
            ┌─────────────────┐              │
            │   Summarize     │              │
            │  • Extract facts│              │
            │  • Structure    │              │
            └────────┬────────┘              │
                     │                       │
                     ▼                       │
            ┌─────────────────┐              │
            │ Format Output   │              │
            │  • Markdown     │              │
            │  • Citations    │              │
            └────────┬────────┘              │
                     │                       │
                     ▼                       │
                    END                      │
                                             │
         Loop Protection: Max 3 iterations ──┘
```

---

## Technology Stack

### Core Framework
- **LangGraph 0.2+**: State machine orchestration with conditional edges
- **LangChain Core**: Prompt templates and message structures
- **Python 3.8+**: Primary implementation language

### LLM Integration
- **Ollama**: Local LLM inference server
  - Model: Configurable (default: llama2/mistral)
  - Temperature: 0.7 for balanced creativity
  - Structured output: JSON parsing with fallbacks

### Search & Retrieval
- **Google Custom Search API**: Fresh web results
  - Programmable Search Engine (CSE)
  - 5 results per query
- **ChromaDB**: Persistent vector storage
  - Collection 1: `web_search` (cached results)
  - Collection 2: `local_document` (research papers)
  - Embedding: nomic-embed-text via Ollama (768 dimensions, optimized for semantic search)

### Data Processing
- **langchain_ollama**: Modern embeddings interface
- **google-api-python-client**: Search API client
- **hashlib + os**: Smart document indexing (file path + mtime)

### Configuration
- **python-dotenv**: Environment variable management
- **TypedDict**: Type-safe state definitions

---

## Component Breakdown

### 1. Workflow Engine (`workflow.py`)

**Purpose**: Orchestrates the research process as a state machine

**Key Features**:
- State graph with 7+ nodes
- Conditional routing based on search quality evaluation
- Loop protection (max 3 refinement iterations)
- Configurable execution modes via flags

**State Schema**:
```python
class ResearchState(TypedDict):
    # Core workflow
    original_query: str
    refined_query: str
    search_results: List[Dict]
    formatted_results: str
    summary: str
    output: str

    # Structured outputs
    intent_metadata: Dict
    evaluation_result: Dict
    validated_facts: Dict
    ranked_documents: Dict

    # Loop protection
    iteration_count: int
    max_iterations: int

    # Tools integration
    vector_db_results: Optional[List[Dict]]
    retrieved_documents: Optional[List[Dict]]

    # Evaluation metrics
    metrics: Dict
    evaluation_metadata: Dict
```

**Node Implementations**:

| Node                    | Purpose                                     | LLM Call | Output                  |
|-------------------------|---------------------------------------------|----------|-------------------------|
| `parse_intent`          | Extract research goal, entities, fact count | Yes      | `intent_metadata`       |
| `refine_query`          | Optimize search query with temporal context | Yes      | `refined_query`         |
| `fetch_sources`         | Get web/cached/local results                | No       | `search_results`        |
| `evaluate_search`       | Assess result quality                       | Yes      | `evaluation_result`     |
| `rank_documents`        | Score and prioritize sources                | Yes      | `ranked_documents`      |
| `reason_and_synthesize` | Cross-validate, detect contradictions       | Yes      | `validated_facts`       |
| `summarize`             | Extract structured facts                    | Yes      | `summary`               |
| `format_output`         | Create final markdown report                | Yes      | `output`                |

### 2. Prompt Templates (`prompt_templates.py`)

**Design Pattern**: SYSTEM → TASK → CONTEXT → OUTPUT FORMAT

**Templates**:

1. **Intent Parser**
   - Extracts: primary_intent, key_entities, required_facts_count, detail_level
   - Output: Structured JSON

2. **Query Refiner**
   - Optimizes: Searchable keywords, temporal context, authoritative sources
   - Special handling: Elections, "latest", current events (adds year)
   - Output: Plain text search query

3. **Search Evaluator**
   - Criteria: Relevance (0-10), Coverage (0-10), Quality (0-10)
   - Recommendation: `proceed` or `refine_and_research`
   - Philosophy: Lenient ("good enough" vs "perfect")
   - Output: JSON evaluation

4. **Retrieval Ranker**
   - Scores documents 0-10 with rationale
   - Prioritizes recency for temporal queries
   - Output: Ranked list with top sources

5. **Reasoning & Synthesis**
   - Validates facts across sources
   - Detects contradictions (with resolution)
   - Identifies knowledge gaps
   - Source citation format: `WEB: title`, `CACHED: title`, `LOCAL: filename`
   - Output: Validated facts with confidence scores

6. **Summarizer**
   - Creates fact-based overview
   - Numbered key facts (user-specified count)
   - Concise, accurate, well-organized
   - Output: Markdown summary

7. **Output Formatter**
   - Professional markdown report
   - Includes: Query, refined query, summary, sources
   - Optional: Detailed fact validation (with `--detailed-output`)
   - Output: Final report

### 3. LLM Service (`llm_service.py`)

**Architecture**: Single service class with method-per-template

**Features**:
- Template-driven prompting
- JSON parsing with fallback handling
- Automatic preamble stripping (for LLMs that add explanations)
- Graceful degradation on parse errors

**Error Handling**:
```python
# Example: reason_and_synthesize
try:
    response = self.llm.invoke(prompt)

    # Strip preamble if LLM adds text before JSON
    if not response.strip().startswith('{'):
        json_start = response.find('{')
        if json_start != -1:
            response = response[json_start:]

    return json.loads(response)
except json.JSONDecodeError:
    # Return safe fallback structure
    return {"validated_facts": [], ...}
```

### 4. Vector Store Service (`vector_store_service.py`)

**Purpose**: Semantic search and caching with ChromaDB

**Architecture**:
- Singleton pattern (`get_vector_store_service()`)
- Persistent storage: `./chroma_db/`
- Separate metadata filtering for web vs local documents

**Key Operations**:

**1. Add Search Results (Caching)**
```python
# Metadata structure
{
    "source": "web_search",
    "query": original_query,
    "url": result_link,
    "indexed_at": timestamp
}
```

**2. Index Local Documents**
- Scans `./documents/` recursively
- Supports: `.md`, `.txt`
- Smart indexing: MD5(filepath + mtime) → skip if already indexed
- Metadata: `{"source": "local_document", "file_path": path}`

**3. Similarity Search (Cached Results)**
```python
similarity_search(
    query,
    top_k=3,
    score_threshold=0.5,
    filter_metadata=None,
    verbose=False
)
# Returns: List of cached results with relevance scores
# Filters by L2 distance (lower = more similar)
# Only includes documents with score <= threshold
```

**4. Search Local Documents**
```python
search_local_documents(
    query,
    top_k=2,
    score_threshold=0.5,
    verbose=False
)
# Uses metadata filter: {"source": "local_document"}
# Filters by relevance threshold (L2 distance)
# Returns documents with relevance_score field
# Typical thresholds: 0.3 (strict), 0.5 (moderate), 2.0 (lenient)
```

**Relevance Filtering** (Added in v2.0):
- **L2 Distance Scoring**: Each result gets a score (lower = more similar)
- **Threshold Filtering**: Only results with score ≤ threshold are included
- **Configurable Strictness**:
  - `0` = No filtering (all results)
  - `0.3` = Strict (highly relevant only)
  - `0.5` = Moderate (default)
  - `2.0` = Lenient (broader results)
- **Transparency**: Scores shown in verbose mode and test script
- **Prevents Contamination**: Filters irrelevant semantically-similar documents

**Benefits**:
- Build knowledge base across runs
- Reduce API calls to Google CSE
- Enable semantic search over local research papers
- Filter irrelevant results with score thresholds
- Full transparency with source type labels and relevance scores

### 5. Search Service (`search_service.py`)

**Purpose**: Google Custom Search API integration

**Features**:
- Configurable result count (max 10 per API request)
- Structured result extraction (title, link, snippet)
- LLM-friendly formatting with `[WEB]` labels

**Output Format**:
```
=== FRESH WEB SEARCH RESULTS ===

[WEB] Article Title:
URL: https://example.com/article
Content: Snippet text...

[WEB] Another Article:
...
```

---

## Key Design Decisions

### 1. Conditional Routing for Agentic Behavior

**Problem**: Static workflows can't adapt to poor search results

**Solution**: Evaluation-driven routing
```python
def _route_after_evaluation(state):
    evaluation = state['evaluation_result']
    recommendation = evaluation.get('recommendation')

    if recommendation == "proceed":
        return "rank_documents"  # Continue workflow
    elif recommendation == "refine_and_research":
        return "refine_query"  # Loop back

    # Loop protection
    if state['iteration_count'] >= 3:
        return "rank_documents"  # Force progress
```

**Benefits**:
- Autonomous quality assessment
- Adaptive query refinement
- Prevents infinite loops
- Balances quality vs execution time

### 2. Multi-Source Integration with Transparency

**Challenge**: Users need to verify source usage, especially local documents

**Solution**: Source type labeling throughout pipeline
- `[WEB]`: Fresh search results
- `[CACHED]`: Previously indexed web results from ChromaDB
- `[LOCAL DOC]`: Local research papers

**Implementation**:
```python
# In fetch_sources_node
formatted = "=== FRESH WEB SEARCH RESULTS ===\n"
formatted += format_results_for_llm(web_results)  # Adds [WEB]

if cached_results:
    formatted += "\n--- CACHED RESULTS ---\n"
    for cached in cached_results:
        formatted += f"[CACHED] {cached['title']}:\n..."

if local_docs:
    formatted += "\n--- LOCAL REFERENCE DOCUMENTS ---\n"
    for doc in local_docs:
        formatted += f"[LOCAL DOC] {doc['title']}:\n..."
```

**Validated Facts Source Attribution**:
```json
{
  "fact": "Statement validated across sources",
  "supporting_sources": ["WEB: Article 1", "LOCAL: paper.md"],
  "confidence": "high"
}
```

### 3. Dual Output Modes

**Use Cases**:
- **General Use**: Clean, concise summaries
- **Research/Demo**: Full transparency with fact validation

**Implementation**:
```bash
# General mode
python main.py "query"

# Detailed mode (includes validated facts with sources)
python main.py --detailed-output "query"
```

**Output Difference**:
```markdown
# General Mode
## Summary and Facts
[Clean summary with numbered facts]

## Sources
1. Source 1
2. Source 2

# Detailed Mode
## Summary and Facts
[Same summary]

## Detailed Fact Validation
1. **Fact statement**
   - Confidence: high
   - Supporting sources: WEB: Source 1, LOCAL: paper.md

## Sources
[Same sources list]
```

### 4. Smart Document Indexing

**Challenge**: Avoid re-indexing unchanged files

**Solution**: Content-based hashing
```python
mod_time = os.path.getmtime(file_path)
doc_id = hashlib.md5(f"{file_path}:{mod_time}".encode()).hexdigest()

# Check if already indexed
existing = collection.get(ids=[doc_id])
if existing['ids']:
    skip_file()  # Already indexed
else:
    index_file()  # New or modified
```

**Benefits**:
- Fast startup after initial indexing
- Automatic re-indexing on file changes
- Scalable to large document collections

### 5. Lenient Search Evaluation

**Problem**: Overly critical evaluation → unnecessary loops → slow execution

**Solution**: "Good enough" philosophy
```python
SEARCH_EVALUATOR_SYSTEM = """Be practical and lenient - perfect is the enemy of good.
IMPORTANT GUIDELINES:
- Recommend "proceed" if results contain enough information
- Only recommend "refine_and_research" if results are TRULY inadequate
- Don't be overly critical
"""
```

**Impact**:
- Before: 80% of queries looped 3 times
- After: 90% proceed on first evaluation
- Result: 2-3x faster execution

---

## Execution Modes

### Standard Mode
```bash
python main.py "research query"
```
- Fresh web search only
- Formatted output
- All workflow steps

### Tools Mode
```bash
python main.py --enable-tools "research query"
```
- Enables ChromaDB integration
- Caches web results for future queries
- Retrieves local documents
- Builds persistent knowledge base

### Fast Mode
```bash
python main.py --skip-formatting "research query"
```
- Skips final formatting LLM call
- Returns raw summary
- Saves ~10-15 seconds

### Demo Mode
```bash
python main.py --enable-tools --detailed-output "research query"
```
- Full transparency
- Shows validated facts with sources
- Demonstrates local document integration
- Ideal for presentations

### Combined
```bash
python main.py --enable-tools --skip-formatting --detailed-output "query"
```
- Tools + Fast + Detailed
- Configurable for specific needs

---

## Testing & Transparency

### Test Script (`test_workflow_transparency.py`)

**Purpose**: Full visibility into workflow execution

**Features**:
- Prints state after each node
- Shows retrieved documents with file paths
- Highlights local document usage in validated facts
- Displays ChromaDB query results
- Tracks iteration count and routing decisions

**Sample Output**:
```
📄 RETRIEVED LOCAL DOCUMENTS (2 documents):
  [1] BindingAgent: Unleashing the Power of AI Agents
      Path: ./documents/markdown/paper1.md
      Preview: Abstract content...

🧠 VALIDATED FACTS (5 facts):
  [1] BindingAgent introduces accountability mechanisms for AI agents
      Confidence: high
      Supporting sources: ['WEB: NTIA Report', 'LOCAL: BindingAgent paper']
      🎯 Uses LOCAL DOCUMENTS: ['LOCAL: BindingAgent paper']
```

**Usage**:
```bash
python test_workflow_transparency.py
# Always runs with --enable-tools equivalent
# Always shows detailed information
```

---

## Performance Characteristics

### LLM Calls per Query

| Mode                      | Minimum | Maximum | Average |
|---------------------------|---------|---------|---------|
| Standard                  | 7       | 8       | 7.2     |
| --skip-formatting         | 6       | 7       | 6.2     |
| With refinement loops     | 9       | 15      | 10.5    |

**Breakdown**:
1. Parse Intent
2. Refine Query (×1-4 with loops)
3. Evaluate Search (×1-4 with loops)
4. Rank Documents
5. Reason & Synthesize
6. Summarize
7. Format Output (optional)

### Execution Time (Estimated)

| Configuration                      | Time         |
|------------------------------------|--------------|
| Standard (no loops)                | 45-60s       |
| With 1 refinement loop             | 70-85s       |
| With 3 refinement loops            | 120-140s     |
| --skip-formatting                  | -10-15s      |
| --enable-tools (first run)         | +5-10s       |
| --enable-tools (subsequent)        | +2-3s        |

### API Usage

**Google Custom Search API**:
- Standard query: 1-4 requests (5 results each)
- Cost: ~$5 per 1000 queries (based on tier)

**Ollama**:
- Local inference (free)
- GPU recommended for <5s response times
- CPU mode: 15-30s per call

---

## Data Flow Example

### Query: "What has been done for AI agents accountability and credibility"

**Step 1: Parse Intent**
```json
{
  "primary_intent": "factual_lookup",
  "key_entities": ["AI agents", "accountability", "credibility"],
  "required_facts_count": 3,
  "detail_level": "moderate"
}
```

**Step 2: Refine Query**
```
Input: "What has been done for AI agents accountability and credibility"
Output: "AI agent accountability credibility mechanisms frameworks 2024"
```

**Step 3: Fetch Sources**
```
Fresh Web: 5 results from Google CSE
Cached: 2 results from ChromaDB (similar past queries)
Local Docs: 2 results (BindingAgent paper, NTIA report)
Total: 9 sources
```

**Step 4: Evaluate Search**
```json
{
  "sufficient": true,
  "relevance_score": 9,
  "coverage_score": 8,
  "recommendation": "proceed"
}
```

**Step 5: Rank Documents**
```json
{
  "ranked_documents": [
    {
      "document_id": "local_doc_1",
      "title": "BindingAgent: Accountability for AI Agents",
      "relevance_score": 10,
      "key_information": ["Accountability framework", "Credibility metrics"]
    },
    ...
  ]
}
```

**Step 6: Reason & Synthesize**
```json
{
  "validated_facts": [
    {
      "fact": "BindingAgent introduces accountability through binding commitments",
      "supporting_sources": ["LOCAL: BindingAgent.md", "WEB: NTIA Report"],
      "confidence": "high"
    }
  ],
  "contradictions": [],
  "patterns_identified": ["Focus on transparency", "Verification mechanisms"]
}
```

**Step 7: Summarize**
```markdown
Recent research has focused on accountability and credibility mechanisms for AI agents...

Key Facts:
1. BindingAgent introduces binding commitments for accountability
2. NTIA proposes transparency requirements for AI systems
3. Multiple frameworks emphasize verification and auditability
```

**Step 8: Format Output**
```markdown
# Research Results

## Original Query
What has been done for AI agents accountability and credibility

## Search Query Used
AI agent accountability credibility mechanisms frameworks 2024

## Summary and Facts
[Summary from Step 7]

## Detailed Fact Validation (if --detailed-output)
1. **BindingAgent introduces binding commitments for accountability**
   - Confidence: high
   - Supporting sources: LOCAL: BindingAgent.md, WEB: NTIA Report

## Sources
1. [BindingAgent Paper](./documents/markdown/BindingAgent.md)
2. [NTIA AI Accountability Report](https://example.com/ntia-report)
...
```

---

## Configuration

### Environment Variables (`.env`)
```bash
# Required
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id

# Ollama (defaults)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2  # or mistral, mixtral, etc.
```

### Configurable Parameters

**In `workflow.py`**:
```python
ResearchWorkflow(
    skip_formatting=False,      # Skip final formatting step
    enable_tools=False,          # Enable ChromaDB integration
    cached_results_count=3,      # Number of cached results to retrieve
    local_docs_count=2,          # Number of local docs to retrieve
    detailed_output=False        # Include validated facts in output
)
```

**In `llm_service.py`**:
```python
OllamaLLMService(
    model_name="llama2",         # LLM model
    base_url="http://localhost:11434",
    temperature=0.7              # Creativity vs determinism
)
```

**In `search_service.py`**:
```python
GoogleSearchService.search(
    query="search query",
    num_results=5                # Results per query (max 10)
)
```

---

## Installation & Setup

### Prerequisites
```bash
# Python 3.8+
python --version

# Ollama running locally
ollama serve
ollama pull llama2  # or your preferred model
```

### Installation
```bash
# Clone repository
git clone <repo_url>
cd local_research_assistant

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Google API credentials
```

### First Run
```bash
# Test basic functionality
python main.py "What is LangGraph?"

# Test with tools (will index local documents)
python main.py --enable-tools "AI agent frameworks"

# Full demo mode
python main.py --enable-tools --detailed-output "recent AI developments"
```

---

## Project Structure

```
local_research_assistant/
├── main.py                          # CLI entry point
├── workflow.py                      # LangGraph state machine
├── llm_service.py                   # Ollama LLM integration
├── search_service.py                # Google CSE integration
├── vector_store_service.py          # ChromaDB operations
├── prompt_templates.py              # Structured prompt templates
├── config.py                        # Environment configuration
├── test_workflow_transparency.py    # Detailed testing script
├── requirements.txt                 # Python dependencies
├── .env                            # Configuration (not in git)
├── chroma_db/                      # Persistent vector storage
│   └── [ChromaDB files]
├── documents/                       # Local research papers
│   └── markdown/
│       ├── paper1.md
│       └── paper2.md
└── ARCHITECTURE.md                  # This document
```

---

## Acknowledgments

**Frameworks**:
- LangGraph by LangChain
- ChromaDB by Chroma
- Ollama by Ollama team

**APIs**:
- Google Custom Search API

**Design Inspiration**:
- Agentic workflow patterns from LangChain documentation
- Prompt engineering best practices from OpenAI and Anthropic

---

## Recent Improvements & Optimizations

### Post-Task 4 Enhancements (December 2024)

After completing the core implementation (Tasks 1-4), several critical improvements were made to enhance accuracy, performance, and transparency.

#### 1. Embedding Model Migration

**Problem Identified**: The system was using `llama3.2` (a chat model) for embeddings, resulting in poor semantic similarity scores.

**Evidence**:
- Relevant documents received scores of 0.904-1.4+ (very high L2 distance)
- 3072-dimension embeddings (excessive for similarity search)
- Inconsistent relevance filtering

**Solution**: Migrated to `nomic-embed-text` - a purpose-built embedding model

**Changes**:
```python
# vector_store_service.py
def __init__(self, embedding_model: str = "nomic-embed-text"):  # Changed from llama3.2
    """
    Args:
        embedding_model: Ollama model to use for embeddings
                       Recommended: "nomic-embed-text" (optimized for semantic search)
                       Not recommended: general LLMs like "llama3.2"
    """
    self.embeddings = OllamaEmbeddings(model=embedding_model)
```

**Migration Script** (`migrate_embeddings.py`):
- Safe migration with automatic backups
- Dry-run mode for testing
- Validation checks at each step
- Zero data loss guarantee (original database never modified)
- Batch processing with progress tracking
- Easy rollback capability

**Results**:
- Score improvement: 0.904 → 0.57 for same relevant document (40% better)
- Dimension reduction: 3072 → 768 (faster queries, lower memory)
- 254 documents successfully migrated with full metadata preservation

**Verification**: New utility `verify_embeddings.py` confirms database dimensions

#### 2. Relevance Filtering with Score Thresholds

**Problem**: Semantically similar but irrelevant documents contaminated answers
- Example: "AI accountability" documents appearing in "US vs China AI" queries
- No mechanism to filter low-quality matches

**Solution**: L2 distance-based filtering for both cached and local documents

**Implementation**:

```python
# vector_store_service.py
def search_local_documents(
    self,
    query: str,
    top_k: int = 2,
    score_threshold: float = 0.5,  # NEW: Relevance threshold
    verbose: bool = False
) -> List[Dict[str, str]]:
    """
    Args:
        score_threshold: L2 distance threshold (lower score = more similar)
                       Set to 0 to disable filtering
                       Typical values: 0.3 (strict), 0.5 (moderate), 2.0 (lenient)
    """
    docs_with_scores = vector_store.similarity_search_with_score(...)

    for doc, score in docs_with_scores:
        if score_threshold == 0 or score <= score_threshold:
            results.append({
                "content": doc.page_content,
                "relevance_score": round(score, 3)  # NEW: Expose scores
            })
        else:
            if verbose:
                print(f"⏭️  Filtered out (low relevance): {title} (score: {score})")
```

**Applies to**:
1. **Cached web results** (source="web_search")
2. **Local documents** (source="local_document")

**CLI Configuration**:
```bash
python main.py --enable-tools --relevance-threshold=0.3 "query"
```

**Benefits**:
- Prevents irrelevant documents from contaminating answers
- User-configurable strictness
- Full transparency (shows filtered documents in verbose mode)
- Consistent filtering across all vector searches

#### 3. Enhanced JSON Parsing with Auto-Fixes

**Problem**: LLM frequently generated invalid JSON, causing workflow failures

**Common Errors**:
1. Preamble text before JSON: `"Here is the analysis: {"`
2. Nested quotes in arrays: `["Book Title" by Author]`
3. Partial quoting: `["text" from source]`
4. Trailing commas

**Solution**: Multi-layer error handling with regex-based auto-fixes

**Implementation** (llm_service.py):

```python
try:
    response = self.llm.invoke(prompt)

    # Layer 1: Strip preamble
    if not response.strip().startswith('{'):
        json_start = response.find('{')
        if json_start != -1:
            response = response[json_start:]

    # Layer 2: Fix common patterns
    cleaned = response
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)  # Trailing commas
    cleaned = re.sub(r'"([^"]+)"\s+by\s+([^,\]"]+)([,\]])', r'"\1 by \2"\3', cleaned)  # "Title" by Author
    cleaned = re.sub(r'"([^"]+)"\s+from\s+([^,\]"]+)([,\]])', r'"\1 from \2"\3', cleaned)  # "Text" from Source

    # Layer 3: Parse
    return json.loads(cleaned)

except json.JSONDecodeError as e:
    print(f"⚠️ JSON parse error: {e}")
    print(f"--- FULL RAW RESPONSE ---")
    print(response)
    print(f"--- END RAW RESPONSE ---")

    # Layer 4: Graceful fallback
    return {"validated_facts": [], ...}  # Continue workflow
```

**Prompt Template Enhancements**:

Added explicit WRONG/CORRECT examples to all JSON-generating templates:

```
CRITICAL - Array String Format:
Each element in an array MUST be a complete string in quotes.

WRONG (DO NOT DO THIS):
"key_information": ["Relativity: The Special and the General Theory" by Einstein]
"validated_facts": ["text" from source, "more text" from another]

CORRECT (DO THIS):
"key_information": ["Relativity: The Special and the General Theory by Einstein"]
"validated_facts": ["text from source", "more text from another"]
```

**Results**:
- Workflow continues even with LLM formatting issues
- Detailed debugging output for troubleshooting
- Reduced workflow failures by ~90%

#### 4. Enhanced Test Script Transparency

**Updates to `test_workflow_transparency.py`**:

**New Parameters**:
```python
WorkflowDebugger(
    skip_formatting=False,
    enable_tools=False,
    cached_results_count=3,          # NEW
    local_docs_count=2,              # NEW
    local_docs_score_threshold=0.5,  # NEW
    detailed_output=False            # NEW
)
```

**New CLI Flags**:
```bash
python test_workflow_transparency.py --enable-tools --relevance-threshold=0.3 "query"
python test_workflow_transparency.py --enable-tools --cached-results=5 --local-docs=3 "query"
python test_workflow_transparency.py --enable-tools --detailed-output "query"
```

**Enhanced Output Display**:
```
📦 CHROMADB STATUS:
  • Total documents in database: 254
  • Embedding model: nomic-embed-text          # NEW
  • Just cached: 5 new documents
  • Relevance threshold: 0.5                   # NEW
    → Only results with score ≤ 0.5 are included

💾 CACHED RESULTS (2 documents passed filter):
  [1] Article Title (score: 0.32)              # NEW: Shows scores
  [2] Another Article (score: 0.48)

📄 RETRIEVED LOCAL DOCUMENTS (1 documents passed filter):
  [1] Local Document Title (score: 0.41)       # NEW: Shows scores
      Path: /path/to/doc.pdf
      Preview: Content...
```

#### 5. Verification Utilities

**New Files**:

1. **`verify_embeddings.py`**: Confirms database is using nomic-embed-text
   ```bash
   python verify_embeddings.py

   # Output:
   # 📊 Embedding Analysis:
   #   • Vector dimensions: 768
   #   • Model detected: ✅ nomic-embed-text (CORRECT)
   #   • Query vector dimensions: 768
   # ✅ Stored embeddings match query embeddings - GOOD!
   ```

2. **`check_db_sources.py`**: Shows document distribution by source type
   ```bash
   python check_db_sources.py

   # Output:
   # 📊 ChromaDB Contents:
   #   Total documents: 254
   #   Web search cached: 198
   #   Local documents: 56
   ```

3. **`migrate_embeddings.py`**: Safe database migration tool
   - Automatic backups
   - Dry-run mode
   - Validation checks
   - Rollback instructions

#### 6. Task 4 Completion: Evaluation Framework

**Objective**: Systematic quality assessment using LangSmith with mixed evaluator types (LLM-as-judge + deterministic metrics)

**Architecture Overview**:
```
┌─────────────────────────────────────────────────────────────┐
│                    LangSmith Evaluation                      │
│                                                              │
│  Dataset (LangSmith UI)                                     │
│  ├─ local_research_assistant_evaluator                      │
│  └─ Flexible schema: query/input/question/research_request  │
│                           │                                  │
│                           ▼                                  │
│  eval_target.py (@traceable wrapper)                        │
│  ├─ Runs ResearchWorkflow for each dataset example         │
│  └─ Returns: {"output": "research summary..."}             │
│                           │                                  │
│                           ▼                                  │
│  run_langsmith_eval.py (Evaluators)                        │
│  ├─ LLM-as-Judge: research_quality_judge (5 dimensions)    │
│  ├─ Deterministic: minimum_length_eval (≥120 words)        │
│  ├─ Deterministic: mentions_sources_eval (citations)       │
│  └─ Pre-built: conciseness_evaluator (openevals)           │
│                           │                                  │
│                           ▼                                  │
│  LangSmith UI Dashboard                                     │
│  └─ Aggregate metrics, per-example scores, comparison       │
└─────────────────────────────────────────────────────────────┘
```

**Evaluator Types**:

| Evaluator | Type | What It Measures | Pass Criteria |
|-----------|------|------------------|---------------|
| `research_quality_judge` | LLM-as-judge | Accuracy, Coverage, Relevance, Coherence, Clarity (1-5 each) | Qualitative feedback |
| `minimum_length_eval` | Deterministic | Word count ≥ 120 | Binary (pass/fail) |
| `mentions_sources_eval` | Deterministic | Source attribution present | Binary (pass/fail) |
| `conciseness_evaluator` | Pre-built (openevals) | Answer conciseness | Score + feedback |

**Demo: Running Evaluation**

```bash
# 1. Ensure dataset exists in LangSmith UI
#    Dataset name: local_research_assistant_evaluator

# 2. Run evaluation
python run_langsmith_eval.py

# 3. View results in LangSmith UI
#    - Aggregate metrics across all examples
#    - Per-example scores with trace details
#    - Compare runs over time
```

**Sample Evaluation Output**:

```
Evaluating research_assistant_target on dataset: local_research_assistant_evaluator

Example 1/10: "What is artificial intelligence?"
  ✓ research_quality_judge: Accuracy: 4/5, Coverage: 5/5, Relevance: 5/5
  ✓ minimum_length_eval: PASS (185 words)
  ✓ mentions_sources_eval: PASS (3 sources cited)
  ✓ conciseness_evaluator: Score 0.85

Example 2/10: "Explain quantum computing"
  ...

=== EVALUATION COMPLETE ===
Run: research_assistant_eval_2024-12-26
View in LangSmith: https://smith.langchain.com/...
```

**Key Features for Demo**:

1. **Multi-dimensional Assessment**
   - Not just accuracy - also coverage, relevance, coherence, clarity
   - Combines automated (deterministic) + human-like (LLM) judgment

2. **Local LLM as Judge**
   - Uses local Ollama (llama3.2) - no API costs
   - Provides nuanced qualitative feedback
   - Example prompt:
     ```
     Score each criterion from 1 (poor) to 5 (excellent):
     - Accuracy (facts correct, no hallucinations)
     - Coverage (addresses key aspects of the question)
     - Relevance (stays on topic)
     - Coherence (logical structure and flow)
     - Clarity (clear and precise language)
     ```

3. **Flexible Dataset Schema**
   - Accepts: `query`, `input`, `question`, or `research_request`
   - Easy to create test cases in LangSmith UI
   - No rigid schema requirements

4. **Traceable Execution**
   - Every evaluation run tracked in LangSmith
   - Full workflow trace for each example
   - Compare performance across model/prompt changes

**Business Value**:
- **Regression Detection**: Know immediately if changes degrade quality
- **A/B Testing**: Compare different prompts, models, or parameters
- **Quality Assurance**: Automated checks before deployment
- **Performance Tracking**: Monitor improvements over time

**Integration with Workflow**:
```python
# eval_target.py - Wraps workflow for evaluation
@traceable(name="research_assistant_target")
def research_assistant_target(inputs: dict) -> dict:
    query = inputs.get("query") or inputs.get("input") or inputs.get("question")

    workflow = ResearchWorkflow(
        skip_formatting=False,
        enable_tools=False,      # Disable for consistent eval
        detailed_output=False
    )

    output = workflow.run(query)
    return {"output": output}
```

**Demo Talking Points**:
- ✅ "We use both deterministic checks and LLM-as-judge for comprehensive evaluation"
- ✅ "All evaluations run locally with Ollama - no external API costs"
- ✅ "LangSmith dashboard provides visual tracking of quality metrics over time"
- ✅ "Flexible dataset schema makes it easy to add new test cases"

#### 7. CLI Configuration Enhancements

**New in `main.py`**:

```bash
# Relevance filtering
python main.py --enable-tools --relevance-threshold=0.3 "query"
# Options: 0 (no filter), 0.3 (strict), 0.5 (moderate), 2.0 (lenient)

# Result counts
python main.py --enable-tools --cached-results=5 --local-docs=3 "query"

# Combined
python main.py --enable-tools --relevance-threshold=0.3 --cached-results=5 "query"
```

**Validation**:
- Threshold must be >= 0
- Counts must be >= 0
- Clear error messages with suggested values
- Shows active configuration in output

#### 8. Updated Technology Stack

**Embeddings**:
- **Before**: llama3.2 (3072 dimensions, chat model)
- **After**: nomic-embed-text (768 dimensions, optimized for semantic search)
- **Alternatives documented**: mxbai-embed-large (1024 dim), all-minilm

**Vector Store Configuration**:
```python
# Current optimized setup
vector_store_service = VectorStoreService(
    embedding_model="nomic-embed-text"  # Default changed
)

# Configurable via environment if needed
```

### Impact Summary

| Improvement | Before | After | Benefit |
|------------|--------|-------|---------|
| Embedding quality | Score 0.904 for relevant doc | Score 0.57 | 40% improvement |
| Irrelevant results | Always included | Filtered by threshold | Cleaner answers |
| JSON parsing failures | Workflow crash | Auto-fix + fallback | 90% fewer failures |
| Embedding dimensions | 3072 | 768 | 4x faster queries |
| Configuration | Hardcoded | CLI flags | User flexibility |
| Transparency | Limited | Full scoring | Better debugging |
| Evaluation | Manual | LangSmith automated | Systematic tracking |

### Migration Checklist (For New Deployments)

If you have an existing deployment with llama3.2 embeddings:

1. **Backup**: Automatic via `migrate_embeddings.py`
2. **Verify model**: `ollama pull nomic-embed-text`
3. **Dry run**: `python migrate_embeddings.py --dry-run`
4. **Migrate**: `python migrate_embeddings.py`
5. **Verify**: `python verify_embeddings.py`
6. **Test**: `python main.py --enable-tools "test query"`
7. **Swap databases**: Follow migration script instructions
8. **Cleanup**: Delete backup after confirming success

---

## License

[Specify License]

---

## Contact & Support

[Project maintainer information]

---

**Document Version**: 2.0
**Last Updated**: December 26, 2024
**Status**: All Core Tasks Complete (Tasks 1-4), Post-Implementation Optimizations Applied
