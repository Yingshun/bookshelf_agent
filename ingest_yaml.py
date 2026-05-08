"""
YAML Document Ingestion for Qdrant

Incrementally embeds YAML API documentation into the Qdrant 'bookshelf_docs' collection.
Each YAML entry (with name, type, function, docstring) becomes one document — no chunking.
Uses MD5 checksums to detect changes and only re-embeds modified/new files.
"""

import os
import json
import hashlib
import yaml
import uuid
import time
import warnings

warnings.filterwarnings('ignore', message='.*Qdrant client version.*')
warnings.filterwarnings('ignore', message='.*Pydantic V1.*')

from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, PointStruct
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

DOCS_PATH = os.getenv("DOCS_PATH", "./docs")
COLLECTION_NAME = "bookshelf_docs"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIMS = 768
CHECKSUM_DIR = "./rag_data"
CHECKSUM_FILE = os.path.join(CHECKSUM_DIR, "checksums.json")
BATCH_SIZE = 100

# ============================================================================
# YAML PARSING
# ============================================================================

def parse_yaml_file(filepath: str) -> list[Document]:
    """Parse a YAML file into LangChain Documents. One document per YAML entry."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    documents = []
    filename = os.path.basename(filepath)

    if data is None:
        print(f"  ⚠ Empty YAML file: {filename}")
        return documents

    if not isinstance(data, list):
        print(f"  ⚠ Expected a list in {filename}, got {type(data).__name__}")
        return documents

    for item in data:
        if not isinstance(item, dict):
            continue

        name = item.get('name', '')
        item_type = item.get('type', '')
        function = item.get('function', '')
        docstring = item.get('docstring', '')
        short_doc = item.get('short_doc', '')

        content = f"{name}\nType: {item_type}\nFunction: {function}\n\n{docstring}"

        metadata = {
            "source": filepath,
            "name": name,
            "type": item_type,
            "short_doc": short_doc,
        }

        documents.append(Document(page_content=content.strip(), metadata=metadata))

    print(f"  → Parsed {len(documents)} document(s) from {filename}")
    return documents


# ============================================================================
# CHECKSUM MANAGEMENT
# ============================================================================

def compute_md5(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def load_checksums() -> dict:
    if os.path.exists(CHECKSUM_FILE):
        with open(CHECKSUM_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_checksums(checksums: dict):
    os.makedirs(CHECKSUM_DIR, exist_ok=True)
    with open(CHECKSUM_FILE, 'w') as f:
        json.dump(checksums, f, indent=2)


# ============================================================================
# QDRANT OPERATIONS
# ============================================================================

def delete_docs_by_source(client: QdrantClient, source: str):
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.source",
                        match=MatchValue(value=source),
                    )
                ]
            ),
        )
        print(f"  → Deleted old entries for: {os.path.basename(source)}")
    except Exception as e:
        print(f"  ⚠ Error deleting entries for {source}: {e}")


# ============================================================================
# MAIN INGESTION
# ============================================================================

def ingest():
    print("\n" + "=" * 60)
    print("YAML Document Ingestion")
    print("=" * 60)

    if not os.path.isdir(DOCS_PATH):
        print(f"\n✗ DOCS_PATH does not exist: {DOCS_PATH}")
        print("  Set DOCS_PATH in .env to point to your YAML documentation directory.")
        return

    yaml_files = []
    for ext in ('*.yaml', '*.yml'):
        yaml_files.extend(Path(DOCS_PATH).rglob(ext))
    yaml_files = sorted(str(f) for f in yaml_files)

    if not yaml_files:
        print(f"\n✗ No .yaml/.yml files found in {DOCS_PATH}")
        return

    print(f"\nFound {len(yaml_files)} YAML file(s) in {DOCS_PATH}")

    current_checksums = {f: compute_md5(f) for f in yaml_files}
    saved_checksums = load_checksums()

    changed_files = [
        f for f in yaml_files
        if f not in saved_checksums or saved_checksums[f] != current_checksums[f]
    ]
    deleted_files = [
        f for f in saved_checksums
        if f not in current_checksums
    ]

    if not changed_files and not deleted_files:
        print("\n✓ All files up to date. Nothing to ingest.")
        run_test_search()
        return

    print(f"\nProcessing {len(changed_files)} changed file(s) and {len(deleted_files)} deleted file(s)")

    client = QdrantClient(url=QDRANT_URL)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIMS, distance=Distance.COSINE),
        )
        print(f"\n✓ Created collection '{COLLECTION_NAME}'")

    for f in changed_files + deleted_files:
        delete_docs_by_source(client, f)

    for f in deleted_files:
        del saved_checksums[f]

    if not changed_files:
        save_checksums({**saved_checksums, **current_checksums})
        print("\n✓ Only deletions processed. Checksums saved.")
        run_test_search()
        return

    print("\nParsing YAML files...")
    all_documents = []
    for filepath in changed_files:
        docs = parse_yaml_file(filepath)
        all_documents.extend(docs)

    if not all_documents:
        print("\n⚠ No documents parsed from changed files.")
        save_checksums({**saved_checksums, **current_checksums})
        return

    total = len(all_documents)
    print(f"\nEmbedding and uploading {total} documents (batch size {BATCH_SIZE})...")
    start_time = time.time()
    uploaded = 0

    for i in range(0, total, BATCH_SIZE):
        batch_docs = all_documents[i:i + BATCH_SIZE]
        texts = [doc.page_content for doc in batch_docs]

        vectors = embeddings.embed_documents(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                },
            )
            for doc, vec in zip(batch_docs, vectors)
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        uploaded += len(batch_docs)

        elapsed = time.time() - start_time
        rate = uploaded / elapsed if elapsed > 0 else 0
        eta = (total - uploaded) / rate if rate > 0 else 0
        print(f"  [{uploaded}/{total}] {rate:.0f} docs/s, ETA {eta:.0f}s")

    elapsed = time.time() - start_time
    print(f"\n✓ Embedded {total} documents in {elapsed:.1f}s")

    updated_checksums = {**saved_checksums, **current_checksums}
    save_checksums(updated_checksums)
    print("✓ Checksums saved.")

    run_test_search()


def run_test_search():
    print(f"\nTest Search")

    client = QdrantClient(url=QDRANT_URL)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if not client.collection_exists(COLLECTION_NAME):
        print("  ✗ Collection does not exist")
        return

    info = client.get_collection(COLLECTION_NAME)
    print(f"  Collection '{COLLECTION_NAME}': {info.points_count} points")

    if info.points_count == 0:
        print("  ⚠ Collection is empty")
        return

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    test_query = "How to configure SR-IOV?"
    print(f'  Query: "{test_query}"')

    results = retriever.invoke(test_query)
    print(f"  Found {len(results)} results:")
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        name = doc.metadata.get("name", "")
        print(f"    {i}. [{os.path.basename(source)}] {name}")


if __name__ == "__main__":
    ingest()
