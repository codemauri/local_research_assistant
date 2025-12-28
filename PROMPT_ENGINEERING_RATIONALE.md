# Prompt Engineering Rationale

## Overview

This document explains the prompt engineering design choices made for the Research Assistant's LangGraph workflow templates. Each template follows structured best practices to maximize LLM performance, consistency, and reliability.

## Core Design Pattern: SYSTEM/TASK/CONTEXT/OUTPUT FORMAT

All templates follow a four-part structure:

### 1. SYSTEM (Role Definition)
**Purpose:** Establishes the AI's persona, capabilities, and behavioral guidelines

**Why it matters:**
- Sets the tone and expertise level for responses
- Defines scope of responsibilities and constraints
- Primes the model for the specific task domain
- Improves consistency across multiple invocations

**Example:**
```python
SYSTEM = """You are an expert intent parser for a research assistant system.

Your role is to analyze user queries and extract:
1. The primary research intent
2. Key entities and concepts
3. Specificity requirements
..."""
```

**Best Practice Applied:** Clear role definition with explicit capabilities list reduces ambiguity and improves task-specific performance.

### 2. TASK (Specific Instruction)
**Purpose:** Provides the concrete task the AI should perform

**Why it matters:**
- Gives clear, actionable instructions
- Includes dynamic context variables (e.g., `{original_query}`)
- Specifies what analysis or transformation is needed
- Reduces interpretation errors

**Example:**
```python
TASK = """Analyze the following research query and extract structured intent information.

Research Query: {original_query}

Extract and output:
1. Primary Intent: (factual_lookup | comparison | ...)
2. Key Entities: List of main subjects/topics
..."""
```

**Best Practice Applied:** Specific, numbered instructions with examples (enumerations) guide the model toward desired output structure.

### 3. CONTEXT (Dynamic Information)
**Purpose:** Provides runtime data needed for the task

**Why it matters:**
- Injects workflow state into the prompt
- Enables data-driven decision making
- Maintains continuity across workflow steps
- Allows templates to be reusable with different inputs

**Example:**
```python
Intent Metadata:
- Primary Intent: {intent_metadata}
- Key Entities: {key_entities}
- Detail Level: {detail_level}
```

**Best Practice Applied:** Structured context presentation (key-value format) makes information scannable and reduces parsing errors.

### 4. OUTPUT FORMAT (Schema Definition)
**Purpose:** Specifies exact output structure expected

**Why it matters:**
- Ensures consistent, parseable responses
- Reduces post-processing complexity
- Enables automated validation
- Improves reliability in production systems

**Example:**
```python
OUTPUT = """Return your analysis in this exact JSON format:
{{
  "primary_intent": "intent_type",
  "key_entities": ["entity1", "entity2"],
  "required_facts_count": number,
  ...
}}"""
```

**Best Practice Applied:** JSON schema with type hints and example values shows the model exactly what structure to produce.

## Template-Specific Design Rationale

### 1. Intent Parser Template

**Design Choices:**
- **Enumerated intent types** (`factual_lookup | comparison | ...`): Constrains output to known categories
- **Multiple extraction targets**: Breaks complex analysis into discrete, manageable parts
- **JSON output**: Enables programmatic routing in conditional workflow edges

**Why these choices:**
- Intent parsing is a classification task - enumerations work better than free-form text
- Structured output enables conditional logic in LangGraph (Task 2)
- Multiple fields capture different dimensions of user intent

### 2. Query Refiner Template

**Design Choices:**
- **"Return ONLY the optimized search query"**: Explicit constraint prevents verbose explanations
- **Principles list in SYSTEM**: Teaches the model what makes a good search query
- **Plain text output**: Search APIs need simple strings, not JSON

**Why these choices:**
- LLMs tend to over-explain - explicit constraints prevent this
- Teaching principles (vs. examples) generalizes better to diverse queries
- Output format matches downstream API requirements

### 3. Search Quality Evaluator Template

**Design Choices:**
- **Multiple scoring dimensions**: Evaluates different quality aspects independently
- **Explicit criteria list**: Provides evaluation framework
- **Recommendation field**: Drives conditional workflow branching

**Why these choices:**
- Multi-dimensional scores provide richer information than binary yes/no
- Explicit criteria reduce subjective interpretation
- Recommendation field enables "refine and re-search" loops (agentic behavior)

### 4. Document Retrieval Ranker Template

**Design Choices:**
- **Relevance scoring (0-10)**: Quantitative ranking enables sorting
- **Rationale requirement**: Forces model to justify rankings
- **Top sources summary**: Provides quick reference for downstream nodes

**Why these choices:**
- Numeric scores enable programmatic sorting and filtering
- Requiring rationale improves ranking quality (chain-of-thought effect)
- Summary field reduces processing load for subsequent steps

### 5. Reasoning and Synthesis Template

**Design Choices:**
- **Six-step reasoning process**: Explicitly models critical thinking workflow
- **Confidence levels**: Captures uncertainty in claims
- **Contradictions tracking**: Identifies conflicts that need resolution

**Why these choices:**
- Structured reasoning process improves analysis quality (ReAct-inspired)
- Confidence levels enable filtering or flagging uncertain facts
- Contradiction tracking surfaced important nuances often lost in summarization

### 6. Summarization Template

**Design Choices:**
- **Two-part structure**: Overview + numbered facts
- **Length constraints**: "1-2 sentences per fact" prevents verbosity
- **Source separation**: Notes that citations will be added later

**Why these choices:**
- Two-part structure matches human reading patterns (skim then detail)
- Length constraints control output size and improve scannability
- Separating citations from facts improves fact clarity

### 7. Output Formatter Template

**Design Choices:**
- **Markdown template**: Shows exact structure expected
- **Professional tone emphasis**: Ensures appropriate formality
- **Clear heading hierarchy**: Organizes information logically

**Why these choices:**
- Template-in-prompt reduces formatting errors
- Tone guidance prevents overly casual or academic extremes
- Hierarchy guidance improves document scannability

## ChatPromptTemplate vs. PromptTemplate

**Choice: ChatPromptTemplate for all templates**

**Rationale:**
- We use Ollama with chat models (llama3.2) that expect role-based messages
- `ChatPromptTemplate` with `SystemMessage` and `HumanMessage` matches model expectations
- Separating system (role) from user (task) messages is best practice for chat models
- Enables future addition of conversation history via `MessagesPlaceholder`

**Alternative considered:** Simple `PromptTemplate` (string-based)
- **Rejected because:** Chat models perform better with role-differentiated messages
- **When it would work:** For completion models (not chat models)

## Prompt Engineering Techniques Applied

### 1. **Role Prompting**
Every template starts with "You are an expert [role]..." to establish persona and expertise level.

**Evidence:** Improves task-specific performance by priming the model's knowledge domain.

### 2. **Few-Shot Learning (Implicit)**
Output format sections include example structures (e.g., JSON schemas with sample values).

**Evidence:** Showing structure is more effective than describing it.

### 3. **Chain of Thought (CoT)**
Reasoning template explicitly lists steps: "Extract → Cross-validate → Identify → Detect → Assess → Form conclusions"

**Evidence:** Structured thinking improves complex reasoning tasks.

### 4. **Constrained Generation**
Enumerations (`factual_lookup | comparison | ...`) and explicit format requirements constrain outputs.

**Evidence:** Reduces hallucination and improves consistency.

### 5. **Separation of Concerns**
Each template handles one workflow step, not multiple responsibilities.

**Evidence:** Single-responsibility prompts are more maintainable and reliable.

## Integration with LangGraph (Task 2)

These templates are designed for LangGraph integration:

### Conditional Edges
- **Search Evaluator** template's `recommendation` field enables branching:
  - `proceed` → Continue to summarization
  - `refine_and_research` → Loop back to query refinement
  - `ask_clarification` → Request user input

### State Management
- All templates accept workflow state variables (e.g., `{original_query}`, `{search_results}`)
- Output formats (JSON) enable state updates for next node
- Structured outputs reduce state transformation complexity

### Node Composition
- Each template maps to a LangGraph node:
  - `intent_parser_template` → `parse_intent` node
  - `query_refiner_template` → `refine_query` node
  - etc.

## Performance Considerations

### Token Efficiency
- **Clear instructions** reduce back-and-forth clarification needs
- **Structured outputs** minimize post-processing token costs
- **Concise system messages** avoid unnecessary prompt bloat

### Reliability
- **Explicit formats** reduce parsing errors
- **Constrained outputs** improve consistency across runs
- **Multi-dimensional evaluation** enables quality checking

### Maintainability
- **Centralized templates** (not scattered in code) enable version control
- **Clear structure** makes updates predictable
- **Documented rationale** aids future modifications

## Testing and Iteration

**Recommended validation approach:**

1. **Unit test each template** with sample inputs
2. **Measure output conformance** to schema
3. **A/B test** variations for quality improvements
4. **Monitor production** performance and iterate

**Metrics to track:**
- Schema conformance rate
- Output quality (human eval sample)
- Downstream node success rate
- End-to-end workflow completion rate

## Future Enhancements

Potential improvements (for later iterations):

1. **Few-shot examples**: Add 1-2 input/output examples per template
2. **Temperature tuning**: Adjust per template (lower for structured tasks, higher for creative)
3. **Prompt versioning**: Track template versions for A/B testing
4. **Dynamic prompts**: Adjust based on query complexity or domain

## References

- **LangChain Prompt Templates Documentation**: https://python.langchain.com/docs/concepts/prompt_templates/
- **Prompt Engineering Best Practices**: https://becomingahacker.org/mastering-prompt-engineering-for-langchain-langgraph-and-ai-agent-applications-e26d85a55f13
- **ReAct Framework**: Reason + Act paradigm for agentic systems
- **Chain of Thought Prompting**: Wei et al., structured reasoning approach

## Sources

- [Prompt Templates | LangChain](https://python.langchain.com/docs/concepts/prompt_templates/)
- [A Guide to Prompt Templates in LangChain | Mirascope](https://mirascope.com/blog/langchain-prompt-template)
- [Mastering Prompt Engineering for LangChain, LangGraph, and AI Agent Applications](https://becomingahacker.org/mastering-prompt-engineering-for-langchain-langgraph-and-ai-agent-applications-e26d85a55f13)
- [Agentic Design Patterns with LangGraph | Towards AI](https://pub.towardsai.net/agentic-design-patterns-with-langgraph-5fe7289187e6)
