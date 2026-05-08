# Agentic RAG Setup Guide

This guide explains how to set up and use the agentic RAG system with YAML API documentation.

## Agentic RAG Flow

```
get_user_context → decide to retrieve or not → [if retrieve: grade relevance → maybe rewrite query] → generate_answer
```
- **LLM decides** whether to retrieve documents
- **Grades** retrieved documents for relevance
- **Rewrites** questions if documents aren't relevant (up to 2 retries)
- Saves tokens and time on simple queries

## File Structure

```
linux-bookshelf-agent/
├── agentic_bookshelf.py        # Agentic RAG pipeline (LangGraph)
├── agentic_chatbot.py          # Gradio web UI
├── agentic_rag_with_mem0.py    # Alternative implementation example
├── ingest_yaml.py              # YAML document ingestion
└── .env                        # Environment configuration
```

## Step 1: Prepare Your YAML Files

Your YAML files should follow this structure:

```yaml
- name: virttest.utils_libvirt.libvirttest()
  type: function
  docstring: |
    Base class for libvirt test utilities.
    
    Provides common functionality for libvirt virtualization tests.

- name: virttest.qemu_devices.QDevice()
  type: class
  docstring: |
    Represents a QEMU device that can be hotplugged.
    
    Attributes:
        driver (str): QEMU driver name
    
    Methods:
        hotplug(): Hotplug the device into a running VM
```

**Key fields:**
- `name`: Function/class name with signature (e.g., `module.class.method()`)
- `type`: Type identifier (`function`, `class`, `method`, etc.)
- `docstring`: Full documentation using YAML multiline (`|`) format

See `example_api_docs.yaml` for a complete example.

## Step 2: Configure Environment

Update your `.env` file:

```bash
# Path to directory containing YAML files
DOCS_PATH=/path/to/your/yaml/docs

# DeepSeek API key for LLM
DEEPSEEK_API_KEY=your-deepseek-api-key
```

## Step 3: Ensure Services Are Running

```bash
# 1. Start Qdrant (if not already running)
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# 2. Start Ollama (if not already running)
ollama serve

# 3. Pull embedding model (first time only)
ollama pull nomic-embed-text
```

## Step 4: Ingest Your YAML Documentation

```bash
python ingest_yaml.py
```

The script scans `DOCS_PATH` for `.yaml`/`.yml` files, uses MD5 checksums for incremental updates, and stores embeddings in the Qdrant `bookshelf_docs` collection. Re-run it after adding or modifying YAML files.

## Step 5: Run the Chatbot

### Option A: Gradio Web UI (Recommended)

```bash
python agentic_chatbot.py
```

Then open: http://localhost:7860

**Special commands:**
- `/memory` - Show memory context from last query
- `/memories` - Show all conversation memories (up to 10)

### Option B: CLI Mode

```bash
python agentic_bookshelf.py
```

Interactive terminal chat with detailed logging of agent decisions.

## How the Agentic Flow Works

### Example 1: Technical Query (Retrieves Docs)

**User:** "How do I use VMXML.new_from_dumpxml()?"

**Agent decisions:**
```
[get_user_context]
  → No memories found for user1

[generate_query_or_respond]
  → Agent decided to retrieve documents

[retrieve]
  → Retrieved 3 documents for query: 'How do I use VMXML.new_from_dumpxml()...'

[grade_documents]
  ✓ Documents are relevant

[generate_answer]
  ✓ Interaction stored in Mem0 for user1
```

**Response:** Uses retrieved docs to give detailed answer with examples

### Example 2: Conversational Query (No Retrieval)

**User:** "Thanks, that was helpful!"

**Agent decisions:**
```
[get_user_context]
  → Found 2 relevant memories for user1

[generate_query_or_respond]
  → Agent responding without retrieval

[generate_answer]
  ✓ Interaction stored in Mem0 for user1
```

**Response:** Responds directly from conversation context

### Example 3: Poor Results (Rewrites Query)

**User:** "How do I do the thing with the bridge?"

**Agent decisions:**
```
[generate_query_or_respond]
  → Agent decided to retrieve documents

[retrieve]
  → Retrieved 3 documents

[grade_documents]
  ✗ Documents not relevant, rewriting question

[rewrite_question]
  → Rewritten: 'How to create a Linux bridge interface?'

[generate_query_or_respond]
  → Agent decided to retrieve documents

[retrieve]
  → Retrieved 3 documents

[grade_documents]
  ✓ Documents are relevant

[generate_answer]
```

**Response:** Better results after query rewrite

## Troubleshooting

### No documents found during search
- Verify documents are ingested into the `bookshelf_docs` collection in Qdrant
- Check collection status: `curl http://localhost:6333/collections/bookshelf_docs`

### Mem0 errors
- Ensure Qdrant is running on `localhost:6333`
- Check that `mem0` collection exists in Qdrant
- Verify DeepSeek API key is valid

### Agent not retrieving documents
- The agent is designed to skip retrieval for conversational queries
- Try a more technical/specific question
- Check console logs to see agent's decision-making

### Ollama connection errors
- Ensure `ollama serve` is running
- Verify `nomic-embed-text` model is pulled

## Architecture Benefits

**Why Agentic RAG is Better:**

1. **Efficiency**: Doesn't retrieve docs for simple questions → faster, cheaper
2. **Quality**: Grades relevance → better answers
3. **Robustness**: Rewrites unclear queries → handles vague questions
4. **Memory**: Uses Mem0 for personalization → remembers user context

**Tradeoffs:**

- More LLM calls (for decision-making, grading, rewriting)
- Slightly more complex to debug
- Requires good prompts for decision nodes
