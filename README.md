# Linux Bookshelf Agent

An agentic RAG (Retrieval-Augmented Generation) chatbot with conversation memory for answering Linux virtualization testing questions. The agent intelligently decides when to retrieve documents, grades their relevance, and rewrites queries for better results.

## Architecture

| Component | Technology |
|-----------|------------|
| **LLM** | DeepSeek (`deepseek-chat`) via LangChain |
| **Embeddings** | Ollama (`nomic-embed-text`, 768-dim) running locally |
| **Vector Store** | Qdrant (`localhost:6333`) |
| **Conversation Memory** | Mem0 with Qdrant backend |
| **Orchestration** | LangGraph (agentic flow with tool-use, grading, query rewriting) |
| **Web UI** | Gradio ChatInterface |
| **Observability** | LangFuse (`localhost:3000`) |
| **Evaluation** | Custom framework (`eval_rag.py` + scikit-learn, sentence-transformers) |

### Agentic RAG Pipeline

```
START → get_user_context → generate_query_or_respond → [tools_condition]
  ├─ (no tool call) → generate_answer → END
  └─ (tool call) → retrieve (top 5) → [grade_documents]
       ├─ (relevant) → generate_answer → END
       └─ (not relevant) → rewrite_question → generate_query_or_respond (max 2 retries)
```

- **LLM decides** whether to retrieve documents (skips retrieval for simple/conversational queries)
- **Grades** retrieved documents for relevance using structured LLM output
- **Rewrites** questions if documents aren't relevant (up to 2 retries)
- **Remembers** conversation history via dual-memory design (short-term + long-term)

### Memory Design

```
User query
  │
  ├─ Short-term: last 10 turns (Gradio session) → immediate context
  ├─ Long-term: Mem0 semantic search (Qdrant) → cross-session knowledge
  │
  └─ Both merged into LLM prompt → generate response
                                      └─ Q&A stored in Mem0
```

## Files

| File | Description |
|------|-------------|
| `agentic_bookshelf.py` | Core agentic RAG pipeline (LangGraph with tool-use, grading, rewriting) |
| `agentic_chatbot.py` | Gradio web UI that wraps `agentic_bookshelf.py` |
| `ingest_yaml.py` | Incremental YAML document ingestion (MD5 checksum-based) |
| `eval_rag.py` | RAG evaluation script (retrieval + answer quality metrics) |
| `eval_test_cases.yaml` | Hand-crafted evaluation test dataset |
| `logging_config.py` | Structured JSON logging configuration |
| `agentic_rag_with_mem0.py` | Alternative implementation example (not used in main app) |

## Prerequisites

- Python 3.13+
- Ollama installed and running (`ollama serve`)
- Qdrant running on `localhost:6333` (see below)
- DeepSeek API key
- (Optional) LangFuse running on `localhost:3000` for tracing

## Setup

1. Start Qdrant (Docker):

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

2. Install dependencies:

```bash
pip install gradio mem0ai qdrant-client litellm langchain langgraph langchain-core langchain-qdrant langchain-ollama python-dotenv langfuse
```

3. Install spaCy model (required by Mem0):

```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

4. Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

5. Create a `.env` file:

```
DOCS_PATH=/path/to/your/yaml/docs
DEEPSEEK_API_KEY=your-deepseek-api-key

# Optional: LangFuse tracing
ENABLE_TRACING=true
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=http://localhost:3000
```

6. Ingest documents into Qdrant:

```bash
python ingest_yaml.py
```

7. Run the chatbot:

```bash
# Gradio web UI (recommended)
python agentic_chatbot.py    # http://localhost:7860

# Or CLI mode
python agentic_bookshelf.py
```

## Usage

- Ask questions about Linux virtualization testing / virttest APIs in the chat interface
- `/memory` — show memory context from last query
- `/memories` — show all conversation memories (up to 10)
- Memory persists across conversations in Qdrant (collection `mem0`)

## Evaluation

The project includes a custom evaluation framework for measuring retrieval and answer quality.

### Quick Start

```bash
# Retrieval-only evaluation (fast, no LLM calls)
python eval_rag.py --skip-answer

# Full evaluation (retrieval + answer quality)
python eval_rag.py

# Filter by category or ID
python eval_rag.py --category api_lookup
python eval_rag.py --ids tc_001 tc_003

# Custom top-K
python eval_rag.py --k 10
```

### Metrics

**Retrieval:**

| Metric | Description |
|--------|-------------|
| P@K | Fraction of retrieved docs that are relevant |
| R@K | Fraction of expected docs that were retrieved |
| MRR | Reciprocal rank of first relevant result (1st=1.0, 2nd=0.5) |
| Hit@K | Whether at least one relevant doc was retrieved (0 or 1) |

**Answer quality:**

| Metric | Description |
|--------|-------------|
| Keyword Hit Rate | Fraction of expected keywords present in the answer |
| Semantic Similarity | Cosine similarity with reference answer (all-MiniLM-L6-v2) |
| Faithfulness | Fraction of technical terms in answer grounded in retrieved docs |

### Test Case Format

Add test cases in `eval_test_cases.yaml`:

```yaml
test_cases:
  - id: tc_001
    question: "What does the vg_check function do?"
    expected_sources:
      - "virttest.staging.lv_utils.vg_check(vg_name)"
    expected_answer_keywords:
      - "volume group"
      - "exists"
    reference_answer: "vg_check checks whether a provided volume group exists."
    category: "api_lookup"
```

### Current Baseline (2026-05-09)

```
k=5, 5 test cases (retrieval only)
Mean P@5=0.20  Mean R@5=1.00  MRR=0.90  Hit Rate=1.00
```

## Observability (LangFuse)

When `ENABLE_TRACING=true` is set in `.env`, all LLM calls and pipeline events are traced to LangFuse.

- Dashboard: `http://localhost:3000`
- Tracing is optional — disabled gracefully if not configured or `langfuse` is not installed

## Known Issues & Optimization Directions

| Issue | Description |
|-------|-------------|
| **Test coverage** | Only 5 direct API lookup test cases; needs fuzzy queries, negative cases, cross-doc questions |
| **Embedding matching** | Compound concept queries (e.g., "logical volume + device mapper") rank lower (RR=0.50 vs 1.00) |
| **Grading window** | `grade_documents()` only uses first 500 chars for relevance grading; may miss key info in longer docs |
| **Duplicate graph execution** | `chat()` runs the graph twice (`stream()` + `invoke()`), wasting API calls |
| **Answer baseline** | Full answer evaluation (keyword/semantic/faithfulness) not yet baselined |
| **Eval memory isolation** | Repeated eval runs accumulate Mem0 data under `eval_user`; may affect consistency |

## Systemd Service

To run as a service, create `/etc/systemd/system/gradio-chatbot.service`:

```ini
[Unit]
Description=Gradio ChatBot Service
After=network.target ollama.service docker.service
Requires=ollama.service

[Service]
Type=simple
User=lizhu
WorkingDirectory=/home/lizhu/linux_bookshelf_agent
Environment="PATH=/usr/bin:/usr/local/bin"
ExecStart=/usr/bin/python3 /home/lizhu/linux_bookshelf_agent/agentic_chatbot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now gradio-chatbot.service
```
