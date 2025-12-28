"""Ollama LLM integration using LangChain with structured prompt templates."""
import json
from typing import Dict, List, Optional

try:
    from langchain_ollama import OllamaLLM
except ImportError:
    # Fallback for different LangChain versions
    from langchain_community.llms import Ollama as OllamaLLM

from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from prompt_templates import get_template


class OllamaLLMService:
    """Service for interacting with Ollama LLM using structured prompt templates."""

    def __init__(self, model_name: str = None, base_url: str = None):
        self.model_name = model_name or OLLAMA_MODEL
        self.base_url = base_url or OLLAMA_BASE_URL
        self.llm = OllamaLLM(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.7
        )

    def parse_intent(self, original_query: str) -> Dict:
        """
        Parse user query to extract structured intent information.

        Args:
            original_query: User's research request

        Returns:
            Dict with intent metadata (primary_intent, key_entities, etc.)
        """
        template = get_template("intent_parser")
        prompt = template.invoke({"original_query": original_query})

        try:
            response = self.llm.invoke(prompt)

            # Strip any preamble before the JSON
            if not response.strip().startswith('{'):
                json_start = response.find('{')
                if json_start != -1:
                    response = response[json_start:]
                    print(f"⚠️ Stripped preamble text before JSON")

            # Parse JSON response
            intent_data = json.loads(response)
            return intent_data
        except json.JSONDecodeError as e:
            # Fallback if JSON parsing fails
            print(f"⚠️ JSON parse error in intent parsing: {e}")
            print(f"\n--- FULL RAW RESPONSE ---")
            print(response)
            print(f"--- END RAW RESPONSE ---\n")

            # Try to fix common JSON issues
            try:
                import re
                cleaned = response

                # Fix 1: Remove trailing commas before ] or }
                cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)

                # Fix 2: Fix array syntax issues
                cleaned = re.sub(r'(\[|,\s*)"([^"]+)"\s+from\s+([^,\]]+)', r'\1"\2 from \3', cleaned)

                intent_data = json.loads(cleaned)
                print(f"✅ Fixed JSON syntax errors")
                return intent_data
            except Exception as fix_error:
                print(f"⚠️ Auto-fix attempt failed: {str(fix_error)[:100]}")
                pass

            # Final fallback
            print(f"⚠️ Using fallback structure")
            return {
                "primary_intent": "factual_lookup",
                "key_entities": [original_query],
                "required_facts_count": 3,
                "detail_level": "moderate",
                "special_requirements": [],
                "suggested_keywords": [original_query]
            }
        except Exception as e:
            raise Exception(f"Error parsing intent: {str(e)}")

    def refine_query(
        self,
        original_query: str,
        intent_metadata: Optional[Dict] = None,
        evaluation_feedback: Optional[Dict] = None
    ) -> str:
        """
        Refine user query into optimized search query.

        Args:
            original_query: User's research request
            intent_metadata: Intent parsing results (optional)
            evaluation_feedback: Feedback from previous search evaluation (for loops)

        Returns:
            Refined search query string
        """
        template = get_template("query_refiner")

        # Prepare metadata
        if intent_metadata is None:
            intent_metadata = {"primary_intent": "unknown"}

        # Build context
        context = {
            "original_query": original_query,
            "intent_metadata": intent_metadata.get("primary_intent", "unknown"),
            "key_entities": str(intent_metadata.get("key_entities", [])),
            "detail_level": intent_metadata.get("detail_level", "moderate")
        }

        prompt = template.invoke(context)

        try:
            response = self.llm.invoke(prompt)
            # Clean up the response
            refined = response.strip().strip('"').strip("'").strip()
            return refined
        except Exception as e:
            raise Exception(f"Error refining query: {str(e)}")

    def evaluate_search_results(
        self,
        original_query: str,
        search_results: str,
        required_facts_count: int = 3
    ) -> Dict:
        """
        Evaluate if search results are sufficient for the query.

        Args:
            original_query: User's research request
            search_results: Formatted search results
            required_facts_count: Number of facts needed

        Returns:
            Dict with evaluation (sufficient, scores, recommendation, etc.)
        """
        template = get_template("search_evaluator")
        prompt = template.invoke({
            "original_query": original_query,
            "search_results": search_results,
            "required_facts_count": required_facts_count
        })

        try:
            response = self.llm.invoke(prompt)
            evaluation = json.loads(response)
            return evaluation
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error in search evaluation: {e}")
            print(f"Raw response: {response[:200]}...")
            # Safe fallback - assume results are acceptable
            return {
                "sufficient": True,
                "relevance_score": 7,
                "coverage_score": 7,
                "quality_score": 7,
                "identified_gaps": [],
                "recommendation": "proceed",
                "suggested_refinement": None
            }
        except Exception as e:
            raise Exception(f"Error evaluating search results: {str(e)}")

    def rank_documents(
        self,
        original_query: str,
        primary_intent: str,
        retrieved_documents: str
    ) -> Dict:
        """
        Rank retrieved documents by relevance.

        Args:
            original_query: User's research request
            primary_intent: Intent type from parsing
            retrieved_documents: Formatted document list

        Returns:
            Dict with ranked documents
        """
        template = get_template("retrieval_ranker")
        prompt = template.invoke({
            "original_query": original_query,
            "primary_intent": primary_intent,
            "retrieved_documents": retrieved_documents
        })

        try:
            response = self.llm.invoke(prompt)

            # Strip any preamble before the JSON
            if not response.strip().startswith('{'):
                json_start = response.find('{')
                if json_start != -1:
                    response = response[json_start:]
                    print(f"⚠️ Stripped preamble text before JSON")

            ranking = json.loads(response)
            return ranking
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error in document ranking: {e}")
            print(f"\n--- FULL RAW RESPONSE ---")
            print(response)
            print(f"--- END RAW RESPONSE ---\n")

            # Try to fix common JSON issues
            try:
                import re
                cleaned = response

                # Fix 1: Remove trailing commas before ] or }
                cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)

                # Fix 2: Fix patterns like ["Title" by author, ...] or ["Title" from source, ...]
                # Match: "quoted text" followed by unquoted text until delimiter
                # Replace with: "quoted text unquoted text"

                # Pattern for "text" by author
                cleaned = re.sub(r'"([^"]+)"\s+by\s+([^,\]"]+)([,\]])', r'"\1 by \2"\3', cleaned)

                # Pattern for "text" from source (already exists in reasoning)
                cleaned = re.sub(r'"([^"]+)"\s+from\s+([^,\]"]+)([,\]])', r'"\1 from \2"\3', cleaned)

                ranking = json.loads(cleaned)
                print(f"✅ Fixed JSON syntax errors")
                return ranking
            except Exception as fix_error:
                print(f"⚠️ Auto-fix attempt failed: {str(fix_error)[:100]}")
                pass

            # Fallback - return documents unranked
            print(f"⚠️ Using fallback structure (empty ranking)")
            return {
                "ranked_documents": [],
                "top_sources": []
            }
        except Exception as e:
            raise Exception(f"Error ranking documents: {str(e)}")

    def reason_and_synthesize(
        self,
        original_query: str,
        formatted_results: str,
        required_facts_count: int = 3
    ) -> Dict:
        """
        Analyze and synthesize information from multiple sources.

        Args:
            original_query: User's research request
            formatted_results: Formatted search/retrieval results
            required_facts_count: Number of facts to extract

        Returns:
            Dict with validated facts, contradictions, patterns, etc.
        """
        template = get_template("reasoning")
        prompt = template.invoke({
            "original_query": original_query,
            "formatted_results": formatted_results,
            "required_facts_count": required_facts_count
        })

        try:
            response = self.llm.invoke(prompt)

            # Strip any preamble before the JSON (fallback for LLMs that add text)
            if not response.strip().startswith('{'):
                # Find the first { and strip everything before it
                json_start = response.find('{')
                if json_start != -1:
                    response = response[json_start:]
                    print(f"⚠️ Stripped preamble text before JSON")

            reasoning = json.loads(response)
            return reasoning
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error in reasoning: {e}")
            print(f"\n--- FULL RAW RESPONSE ---")
            print(response)
            print(f"--- END RAW RESPONSE ---\n")

            # Try to fix common JSON issues
            try:
                import re
                cleaned = response

                # Fix 1: Remove trailing commas before ] or }
                cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)

                # Fix 2: Fix array syntax like ["text" from source, "text2" from source2]
                # This handles patterns where quotes are only around part of the string
                # Pattern: "text" from source -> text from source (remove inner quotes)
                cleaned = re.sub(r'(\[|,\s*)"([^"]+)"\s+from\s+([^,\]]+)', r'\1"\2 from \3', cleaned)

                reasoning = json.loads(cleaned)
                print(f"✅ Fixed JSON syntax errors")
                return reasoning
            except Exception as fix_error:
                print(f"⚠️ Auto-fix attempt failed: {str(fix_error)[:100]}")
                pass

            # Fallback - create basic structure
            print(f"⚠️ Using fallback structure (empty facts)")
            return {
                "validated_facts": [],
                "contradictions": [],
                "patterns_identified": [],
                "knowledge_gaps": []
            }
        except Exception as e:
            raise Exception(f"Error in reasoning and synthesis: {str(e)}")

    def summarize_results(
        self,
        original_query: str,
        validated_facts: Dict,
        formatted_results: str,
        num_facts: int = 3
    ) -> str:
        """
        Create structured summary from validated information.

        Args:
            original_query: User's research request
            validated_facts: Output from reasoning step
            formatted_results: Original search results for context
            num_facts: Number of facts to include

        Returns:
            Formatted summary string
        """
        template = get_template("summarizer")

        # Convert validated_facts to string for template
        facts_str = json.dumps(validated_facts, indent=2)

        prompt = template.invoke({
            "original_query": original_query,
            "num_facts": num_facts,
            "validated_facts": facts_str,
            "formatted_results": formatted_results
        })

        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            raise Exception(f"Error generating summary: {str(e)}")

    def format_output(
        self,
        original_query: str,
        refined_query: str,
        summary: str,
        sources: List[Dict],
        validated_facts: Optional[Dict] = None
    ) -> str:
        """
        Format final research report.

        Args:
            original_query: User's original request
            refined_query: Search query used
            summary: Generated summary
            sources: List of source documents
            validated_facts: Optional validated facts with confidence (for detailed output)

        Returns:
            Formatted markdown report
        """
        template = get_template("output_formatter")

        # Format sources as string
        sources_str = "\n".join([
            f"{i+1}. {src.get('title', 'Untitled')} - {src.get('link', '')}"
            for i, src in enumerate(sources)
        ])

        # Format validated facts if provided (detailed output mode)
        validated_facts_str = ""
        if validated_facts and validated_facts.get('validated_facts'):
            validated_facts_str = "\n\n## Detailed Fact Validation\n\n"
            for i, fact in enumerate(validated_facts['validated_facts'], 1):
                validated_facts_str += f"{i}. **{fact.get('fact', 'N/A')}**\n"
                validated_facts_str += f"   - Confidence: {fact.get('confidence', 'N/A')}\n"
                supporting = fact.get('supporting_sources', [])
                if supporting:
                    validated_facts_str += f"   - Supporting sources: {', '.join(supporting)}\n"
                validated_facts_str += "\n"

        prompt = template.invoke({
            "original_query": original_query,
            "refined_query": refined_query,
            "summary": summary,
            "sources": sources_str,
            "validated_facts": validated_facts_str
        })

        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            raise Exception(f"Error formatting output: {str(e)}")
