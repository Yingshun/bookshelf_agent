# Linux Bookshelf Agent

An agentic RAG (Retrieval-Augmented Generation) chatbot with conversation memory for answering Linux virtualization testing questions. The agent intelligently decides when to retrieve documents, grades their relevance, and rewrites queries for better results.

## Architecture

- **LLM**: DeepSeek (`deepseek-chat`) via LangChain
- **Embeddings**: Ollama (`nomic-embed-text`) running locally
- **Vector Store**: Qdrant (`localhost:6333`, collection `bookshelf_docs`)
- **Conversation Memory**: Mem0 with Qdrant backend (`localhost:6333`, collection `mem0`)
- **Orchestration**: LangGraph (agentic flow with tool-use, grading, and query rewriting)
- **Web UI**: Gradio ChatInterface

### Agentic RAG Flow

```
get_user_context → decide to retrieve or not → [if retrieve: grade relevance → maybe rewrite query] → generate_answer
```

- **LLM decides** whether to retrieve documents (skips retrieval for simple/conversational queries)
- **Grades** retrieved documents for relevance
- **Rewrites** questions if documents aren't relevant (up to 2 retries)
- **Remembers** conversation history via Mem0

## Files

| File | Description |
|------|-------------|
| `agentic_bookshelf.py` | Core agentic RAG pipeline (LangGraph with tool-use, grading, rewriting) |
| `agentic_chatbot.py` | Gradio web UI that wraps `agentic_bookshelf.py` |
| `agentic_rag_with_mem0.py` | Alternative implementation example (uses OpenAI embeddings + in-memory vector store) |
| `AGENTIC_RAG_SETUP.md` | Detailed setup and usage guide |

## Prerequisites

- Python 3.13+
- Ollama installed and running (`ollama serve`)
- Qdrant running on `localhost:6333` (see below)
- DeepSeek API key

## Setup

1. Start Qdrant (Docker):

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

2. Install dependencies:

```bash
pip install gradio mem0ai qdrant-client litellm langchain langgraph langchain-core langchain-qdrant langchain-ollama python-dotenv
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
```

6. Ingest your documents into Qdrant (documents should already be in the vector store, or use an external ingestion tool).

7. Run the chatbot:

```bash
python agentic_chatbot.py
```

Open http://localhost:7860 in your browser.

Or run in CLI mode:

```bash
python agentic_bookshelf.py
```

## Usage

- Ask questions about Linux virtualization testing / virttest APIs in the chat interface
- `/memory` — show memory context from last query
- `/memories` — show all conversation memories (up to 10)
- Memory persists across conversations in Qdrant (collection `mem0`)

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
