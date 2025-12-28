"""Check what types of documents are in ChromaDB."""
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

db = Chroma(
    persist_directory='./chroma_db',
    embedding_function=OllamaEmbeddings(model='nomic-embed-text'),
    collection_name='research_cache'
)

docs = db.get()
sources = [m.get('source', 'unknown') for m in docs['metadatas']]

print(f"📊 ChromaDB Contents:")
print(f"   Total documents: {len(docs['ids'])}")
print(f"   Web search cached: {sources.count('web_search')}")
print(f"   Local documents: {sources.count('local_document')}")
print(f"   Unknown: {sources.count('unknown')}")
