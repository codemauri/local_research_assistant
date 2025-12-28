"""
Vector Store Service
====================

Provides semantic search over cached research results using vector embeddings.
Designed with a clean abstraction to easily swap between backends.

Current Implementation: ChromaDB Persistent
  - Saves to ./chroma_db directory
  - Builds knowledge base over time across runs
  - Can search previous research results semantically

Easy to swap to: InMemoryVectorStore, ChromaDB in-memory, FAISS, Pinecone, etc.
"""

from typing import List, Dict, Optional
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


class VectorStoreService:
    """
    Vector store service with swappable backend.

    Current: ChromaDB Persistent (builds knowledge base over time)

    Easy swap options:
    - InMemoryVectorStore (zero dependencies, ephemeral)
    - ChromaDB in-memory (ephemeral, like InMemory)
    - FAISS (high performance CPU)
    - Pinecone (cloud-based)
    """

    def __init__(self, embedding_model: str = "nomic-embed-text"):
        """
        Initialize vector store service.

        Args:
            embedding_model: Ollama model to use for embeddings
                           Recommended: "nomic-embed-text" (optimized for semantic search)
                           Alternative: "mxbai-embed-large", "all-minilm"
                           Not recommended: general LLMs like "llama3.2"
        """
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vector_store: Optional[VectorStore] = None

    def _get_vector_store(self) -> VectorStore:
        """
        Get or create the vector store.

        ⚡ SWAP POINT: Change this method to use a different backend.

        Current: ChromaDB Persistent
        ✅ Builds knowledge base over time
        ✅ Saves to ./chroma_db directory
        ✅ Good M1 performance (ARM-optimized)
        ✅ Enables semantic search across all past queries
        """
        if self.vector_store is None:
            # OPTION 2B (CURRENT): ChromaDB Persistent - Builds knowledge base over time
            from langchain_chroma import Chroma
            self.vector_store = Chroma(
                persist_directory="./chroma_db",
                embedding_function=self.embeddings,
                collection_name="research_cache"
            )

            # ============================================================
            # OTHER OPTIONS (comment out OPTION 2B above to use these)
            # ============================================================

            # OPTION 1: InMemoryVectorStore - Zero dependencies
            # from langchain_core.vectorstores import InMemoryVectorStore
            # self.vector_store = InMemoryVectorStore(self.embeddings)

            # OPTION 2A: ChromaDB In-Memory (ephemeral, no persistence)
            # self.vector_store = Chroma(
            #     embedding_function=self.embeddings,
            #     collection_name="research_cache"
            # )

            # OPTION 3: FAISS (high performance, no persistence)
            # Dependency: pip install faiss-cpu
            # from langchain_community.vectorstores import FAISS
            # self.vector_store = FAISS.from_texts(
            #     ["placeholder"],
            #     self.embeddings,
            #     metadatas=[{"source": "init"}]
            # )

            # OPTION 4: Pinecone (cloud-based, requires API key)
            # Dependency: pip install pinecone-client
            # from langchain_community.vectorstores import Pinecone
            # self.vector_store = Pinecone.from_existing_index(
            #     index_name="research-assistant",
            #     embedding=self.embeddings
            # )

        return self.vector_store

    def get_collection_count(self) -> int:
        """Get the total number of documents in the vector store."""
        vector_store = self._get_vector_store()
        try:
            # ChromaDB-specific method
            if hasattr(vector_store, '_collection'):
                return vector_store._collection.count()
            return 0
        except:
            return 0

    def add_search_results(self, search_results: List[Dict[str, str]], verbose: bool = False) -> int:
        """
        Add search results to the vector store.

        Args:
            search_results: List of search result dicts with title, link, snippet
            verbose: If True, print detailed information about what's being stored

        Returns:
            Number of documents added
        """
        if not search_results:
            return 0

        # Get count before adding
        count_before = self.get_collection_count() if verbose else 0

        # Convert search results to Document objects
        documents = []
        for result in search_results:
            content = f"{result.get('title', '')}\n{result.get('snippet', '')}"
            metadata = {
                "title": result.get('title', ''),
                "link": result.get('link', ''),
                "source": "web_search"
            }
            documents.append(Document(page_content=content, metadata=metadata))

        # Add to vector store
        vector_store = self._get_vector_store()
        vector_store.add_documents(documents)

        if verbose:
            count_after = self.get_collection_count()
            print(f"   📦 ChromaDB Status:")
            print(f"      • Documents before: {count_before}")
            print(f"      • Documents added: {len(documents)}")
            print(f"      • Total documents: {count_after}")
            print(f"   📝 Stored documents:")
            for i, doc in enumerate(documents, 1):
                title = doc.metadata.get('title', 'Untitled')[:70]
                print(f"      [{i}] {title}")

        return len(documents)

    def similarity_search(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.5,
        filter_metadata: Optional[Dict] = None,
        verbose: bool = False
    ) -> List[Dict[str, str]]:
        """
        Search for semantically similar content with relevance filtering.

        Args:
            query: Search query
            top_k: Maximum number of results to return
            score_threshold: L2 distance threshold for filtering (lower score = more similar)
                           Set to 0 to disable filtering (include all documents)
                           Typical values: 0.3 (strict), 0.5 (moderate), 2.0 (lenient)
                           Default: 0.5
            filter_metadata: Optional metadata filters (if backend supports it)
            verbose: If True, print detailed information about the search

        Returns:
            List of similar results with content and metadata (may be fewer than top_k if filtered)
        """
        vector_store = self._get_vector_store()

        if verbose:
            total_docs = self.get_collection_count()
            print(f"   🔍 Querying ChromaDB:")
            print(f"      • Query: {query}")
            print(f"      • Top K: {top_k}")
            print(f"      • Score threshold: {score_threshold}")
            print(f"      • Total docs in DB: {total_docs}")

        # Perform similarity search with scores
        if filter_metadata:
            docs_with_scores = vector_store.similarity_search_with_score(
                query, k=top_k * 2, filter=filter_metadata
            )
        else:
            docs_with_scores = vector_store.similarity_search_with_score(query, k=top_k * 2)

        # Filter by relevance score and convert to dict format
        results = []
        for doc, score in docs_with_scores:
            # Only include documents that meet the relevance threshold
            # Note: score_threshold = 0 disables filtering (includes all)
            if score_threshold == 0 or score <= score_threshold:
                results.append({
                    "content": doc.page_content,
                    "title": doc.metadata.get("title", ""),
                    "link": doc.metadata.get("link", ""),
                    "source": doc.metadata.get("source", ""),
                    "relevance_score": round(score, 3)
                })
            else:
                if verbose:
                    title = doc.metadata.get("title", "Untitled")
                    print(f"      ⏭️  Filtered out (low relevance): {title[:60]} (score: {round(score, 3)})")

            # Stop once we have enough results
            if len(results) >= top_k:
                break

        if verbose:
            print(f"   📊 Relevant results found: {len(results)}")
            for i, result in enumerate(results, 1):
                title = result.get('title', 'Untitled')[:70]
                score = result.get('relevance_score', 'N/A')
                print(f"      [{i}] {title} (score: {score})")

        return results

    def index_local_documents(self, documents_dir: str = "./documents", verbose: bool = False) -> int:
        """
        Index local documents from a directory into the vector store.

        Tracks indexed documents to avoid re-indexing unchanged files.

        Args:
            documents_dir: Directory containing documents to index
            verbose: If True, print detailed information about indexing

        Returns:
            Number of new documents indexed
        """
        import os
        import hashlib

        if not os.path.exists(documents_dir):
            if verbose:
                print(f"   📁 Documents directory not found: {documents_dir}")
            return 0

        # Get all .txt and .md files (recursively search subdirectories)
        allowed_extensions = {'.txt', '.md'}
        doc_files = []
        for root, dirs, files in os.walk(documents_dir):
            for filename in files:
                # Skip README files
                if filename == 'README.md':
                    continue
                ext = os.path.splitext(filename)[1].lower()
                if ext in allowed_extensions:
                    doc_files.append(os.path.join(root, filename))

        if not doc_files:
            if verbose:
                print(f"   📁 No documents found in {documents_dir}")
            return 0

        vector_store = self._get_vector_store()
        new_docs_count = 0

        # Get already indexed document IDs (we'll use file path hash as ID)
        indexed_ids = set()
        try:
            if hasattr(vector_store, '_collection'):
                existing = vector_store._collection.get()
                if existing and 'metadatas' in existing:
                    for metadata in existing['metadatas']:
                        if metadata and 'doc_id' in metadata:
                            indexed_ids.add(metadata['doc_id'])
        except:
            pass  # If we can't get existing docs, just index all

        if verbose:
            print(f"   📁 Scanning {len(doc_files)} local documents...")

        for doc_path in doc_files:
            # Create document ID from file path and modification time
            mod_time = os.path.getmtime(doc_path)
            doc_id = hashlib.md5(f"{doc_path}:{mod_time}".encode()).hexdigest()

            # Skip if already indexed
            if doc_id in indexed_ids:
                if verbose:
                    filename = os.path.basename(doc_path)
                    print(f"      ⏭️  Skipped (already indexed): {filename}")
                continue

            # Read and index document
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Limit content length
                max_length = 2000
                if len(content) > max_length:
                    content = content[:max_length] + f"\n[Truncated - {len(content)} chars total]"

                # Create document with metadata
                filename = os.path.basename(doc_path)
                doc = Document(
                    page_content=content,
                    metadata={
                        "title": filename,
                        "source": "local_document",
                        "doc_id": doc_id,
                        "file_path": doc_path
                    }
                )

                vector_store.add_documents([doc])
                new_docs_count += 1

                if verbose:
                    print(f"      ✅ Indexed: {filename} ({len(content)} chars)")

            except Exception as e:
                if verbose:
                    print(f"      ❌ Error indexing {doc_path}: {e}")

        if verbose and new_docs_count > 0:
            print(f"   📦 Indexed {new_docs_count} new local documents")

        return new_docs_count

    def search_local_documents(
        self,
        query: str,
        top_k: int = 2,
        score_threshold: float = 0.5,
        verbose: bool = False
    ) -> List[Dict[str, str]]:
        """
        Search for semantically similar local documents with relevance filtering.

        Args:
            query: Search query
            top_k: Maximum number of results to return
            score_threshold: L2 distance threshold for filtering (lower score = more similar)
                           Only documents with L2 distance <= threshold are included
                           Set to 0 to disable filtering (include all documents)
                           Typical values: 0.3 (strict), 0.5 (moderate), 2.0 (lenient)
                           Default: 0.5 (moderate relevance filtering)
            verbose: If True, print detailed information

        Returns:
            List of relevant local documents (may be fewer than top_k if filtered)
        """
        vector_store = self._get_vector_store()

        if verbose:
            print(f"   📄 Querying local documents:")
            print(f"      • Query: {query}")
            print(f"      • Top K: {top_k}")
            print(f"      • Score threshold: {score_threshold}")

        # Search with filter for local documents only and get similarity scores
        try:
            docs_with_scores = vector_store.similarity_search_with_score(
                query,
                k=top_k * 2,  # Get more candidates to filter
                filter={"source": "local_document"}
            )
        except:
            # If filtering fails, fall back to unfiltered search and filter manually
            all_docs_with_scores = vector_store.similarity_search_with_score(query, k=top_k * 5)
            docs_with_scores = [(d, s) for d, s in all_docs_with_scores
                               if d.metadata.get('source') == 'local_document']

        # Filter by relevance score and convert to dict format
        results = []
        for doc, score in docs_with_scores:
            # Only include documents that meet the relevance threshold
            # Note: score_threshold = 0 disables filtering (includes all)
            if score_threshold == 0 or score <= score_threshold:
                results.append({
                    "content": doc.page_content,
                    "title": doc.metadata.get("title", ""),
                    "source": doc.metadata.get("source", ""),
                    "file_path": doc.metadata.get("file_path", ""),
                    "relevance_score": round(score, 3)
                })
            else:
                if verbose:
                    title = doc.metadata.get("title", "Untitled")
                    print(f"      ⏭️  Filtered out (low relevance): {title} (score: {round(score, 3)})")

            # Stop once we have enough results
            if len(results) >= top_k:
                break

        if verbose:
            print(f"   📊 Relevant local documents found: {len(results)}")
            for i, result in enumerate(results, 1):
                title = result.get('title', 'Untitled')
                score = result.get('relevance_score', 'N/A')
                print(f"      [{i}] {title} (score: {score})")

        return results

    def clear(self) -> None:
        """Clear the vector store (useful for testing or cleanup)."""
        self.vector_store = None


# Singleton instance
_vector_store_service: Optional[VectorStoreService] = None


def get_vector_store_service() -> VectorStoreService:
    """Get the global vector store service instance."""
    global _vector_store_service
    if _vector_store_service is None:
        _vector_store_service = VectorStoreService()
    return _vector_store_service
