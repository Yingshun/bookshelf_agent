import hashlib
import json
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

load_dotenv()

# Create embedder
embeddings = OllamaEmbeddings(model='nomic-embed-text')

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

docs_path = os.getenv("DOCS_PATH")

def compute_checksums(doc_path):
    checksums = {}
    for root, dirs, files in os.walk(doc_path):
        for f in files:
            if f.endswith(".md"):
                filepath = os.path.join(root, f)
                with open(filepath, "rb") as fh:
                    checksums[filepath] = hashlib.md5(fh.read()).hexdigest()
    return checksums

def load_saved_checksums(path='./rag_data/checksums.json'):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def save_checksums(checksums, path='./rag_data/checksums.json'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(checksums, f)

current_checksums = compute_checksums(docs_path)
saved_checksums = load_saved_checksums()

changed = [f for f in current_checksums if current_checksums[f] != saved_checksums.get(f)]
deleted = [f for f in saved_checksums if f not in current_checksums]

# Initialize vectorstore here so it's available for the test search regardless of whether docs changed
client = QdrantClient(url="http://localhost:6333")
vectorstore = QdrantVectorStore(
    client=client,
    collection_name="bookshelf_docs",
    embedding=embeddings,
)

if not changed and not deleted:
    print("No document changed. Skipping embedding.")
else:
    print(f"Changed files: {changed}")
    print(f"Deleted files: {deleted}")

    # Remove old chunks from changed/deleted files
    for filepath in changed + deleted:
        client.delete(
            collection_name="bookshelf_docs",
            points_selector=Filter(
                must=[FieldCondition(key="metadata.source", match=MatchValue(value=filepath))]
            ),
        )
        print(f"Removed old chunks for {filepath}")
    
    # Load and embed only changed files
    if changed:
        new_docs = []
        for filepath in changed:
            new_docs.extend(TextLoader(filepath).load())

        new_chunks = splitter.split_documents(new_docs)
        vectorstore.add_documents(new_chunks)
        print(f"Embedded {len(new_chunks)} new chunks")

    save_checksums(current_checksums)
    print("Checksums saved.")

# Test a search
results = vectorstore.similarity_search("How to configure SR-IOV?", k=3)
print(f'\nTest search for "How to configure SR-IOV?" returned {len(results)} results:')
for i, doc in enumerate(results):
    print(f"\n--- Result {i} (source: {doc.metadata['source']}) ---")
    print(doc.page_content[:300])
