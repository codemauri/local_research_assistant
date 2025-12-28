"""
Verify Embedding Model in ChromaDB
===================================

Checks which embedding model is being used by inspecting vector dimensions.

Expected dimensions:
- nomic-embed-text: 768
- llama3.2: 3072
- mxbai-embed-large: 1024
"""

import sys
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def verify_embeddings():
    """Verify which embedding model is in use."""
    print("🔍 Verifying ChromaDB Embedding Model...\n")

    try:
        # Load database with nomic-embed-text (current default)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        db = Chroma(
            persist_directory='./chroma_db',
            embedding_function=embeddings,
            collection_name='research_cache'
        )

        # Get sample document to check dimensions
        docs = db.get(limit=1, include=['embeddings'])

        if not docs or not docs.get('ids'):
            print("❌ No documents found in database")
            return

        # Check embedding dimensions
        if docs.get('embeddings') is not None and len(docs['embeddings']) > 0:
            embedding = docs['embeddings'][0]
            dimensions = len(embedding)

            print(f"📊 Embedding Analysis:")
            print(f"  • Vector dimensions: {dimensions}")

            # Identify model
            if dimensions == 768:
                print(f"  • Model detected: ✅ nomic-embed-text (CORRECT)")
                print(f"  • Status: Using optimized embedding model")
            elif dimensions == 3072:
                print(f"  • Model detected: ⚠️  llama3.2 (OLD)")
                print(f"  • Status: Migration needed - run migrate_embeddings.py")
            elif dimensions == 1024:
                print(f"  • Model detected: mxbai-embed-large")
                print(f"  • Status: Alternative embedding model in use")
            else:
                print(f"  • Model detected: Unknown ({dimensions} dimensions)")

            # Show total documents
            total_docs = len(db.get()['ids'])
            print(f"\n  • Total documents: {total_docs}")

            # Show embedding model being used for queries
            print(f"\n  • Query embedding model: {embeddings.model}")
            query_dims = len(embeddings.embed_query("test"))
            print(f"  • Query vector dimensions: {query_dims}")

            if dimensions == query_dims:
                print(f"\n✅ Stored embeddings match query embeddings - GOOD!")
            else:
                print(f"\n❌ DIMENSION MISMATCH!")
                print(f"   Stored: {dimensions}, Query: {query_dims}")
                print(f"   This will cause errors. Run migration.")

        else:
            print("⚠️  Could not retrieve embeddings from database")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    verify_embeddings()
