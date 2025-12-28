# Local Documents Directory

This directory is for storing local reference documents that will be indexed into ChromaDB and made available during research queries.

## Supported Formats
- `.txt` - Plain text files
- `.md` - Markdown files

## How It Works
1. Place your documents in this directory
2. On next run with `--enable-tools`, documents will be automatically indexed into ChromaDB
3. During research, semantically relevant local documents will be included in the analysis
4. Documents are only indexed once (tracked to avoid re-indexing)

## Example Use Cases
- Research papers (converted to text)
- Company guidelines or documentation
- Personal notes and summaries
- Reference materials on specific topics
- Curated knowledge base

## Example
```bash
# Add a document
echo "Python is great for data science and machine learning." > python_notes.txt

# Run research with tools enabled
python main.py --enable-tools "Why use Python for data science?"

# The assistant will find and use python_notes.txt if relevant
```

## Notes
- Documents are indexed based on semantic similarity
- Only relevant documents are included in each query
- If no relevant documents exist, workflow continues with web results only
- Maximum content length per document: 2000 characters
