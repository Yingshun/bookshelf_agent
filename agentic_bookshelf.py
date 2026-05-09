"""
Agentic RAG with Mem0 for Linux Bookshelf
- Uses Qdrant (persistent) + Ollama embeddings (local)
- LLM decides when to retrieve documents
- Grades document relevance
- Rewrites queries if needed
"""

import warnings
warnings.filterwarnings('ignore', message='.*Qdrant client version.*')
warnings.filterwarnings('ignore', message='.*Pydantic V1.*')

from typing import Annotated, TypedDict, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from mem0 import Memory
from pydantic import BaseModel, Field
import os
import time
from dotenv import load_dotenv
from logging_config import get_logger

logger = get_logger("pipeline")

load_dotenv()

# ============================================================================
# LANGFUSE TRACING (optional, enabled via ENABLE_TRACING=true)
# ============================================================================

langfuse_handler = None
if os.environ.get("ENABLE_TRACING", "").lower() == "true":
    try:
        from langfuse.langchain import CallbackHandler
        langfuse_handler = CallbackHandler()
        logger.info("LangFuse tracing enabled")
    except ImportError:
        logger.warning("langfuse package not installed, tracing disabled")
    except Exception as e:
        logger.warning("LangFuse init failed: %s", e)

langfuse_callbacks = [langfuse_handler] if langfuse_handler else []

# ============================================================================
# CONFIGURATION
# ============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    mem0_user_id: str
    user_context: str
    retry_count: int

class GradeDocuments(BaseModel):
    """Schema for document relevance grading"""
    binary_score: Literal["yes", "no"] = Field(
        description="Relevance score: 'yes' if relevant, 'no' if not"
    )

# Initialize LLM (DeepSeek)
llm = init_chat_model(
    "deepseek-chat",
    model_provider="deepseek",
    temperature=0
)

# Initialize Mem0 with Qdrant backend
mem0_config = {
    "llm": {
        "provider": "litellm",
        "config": {
            "model": "deepseek/deepseek-chat",
            "temperature": 0,
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "mem0",
            "url": "http://localhost:6333",
            "embedding_model_dims": 768,
        }
    }
}

mem0 = Memory.from_config(mem0_config)

# Initialize Qdrant vector store for documents
embeddings = OllamaEmbeddings(model='nomic-embed-text')
qdrant_client = QdrantClient(url="http://localhost:6333")

if not qdrant_client.collection_exists("bookshelf_docs"):
    from qdrant_client.models import Distance, VectorParams
    qdrant_client.create_collection(
        collection_name="bookshelf_docs",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name="bookshelf_docs",
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ============================================================================
# TOOL DEFINITION
# ============================================================================

@tool
def retrieve_documents(query: str) -> str:
    """Search the knowledge base for relevant Linux documentation and technical information."""
    t0 = time.perf_counter()
    docs = retriever.invoke(query)
    elapsed = time.perf_counter() - t0

    logger.info("Retrieved %d documents for query: '%s...'",
                len(docs), query[:50],
                extra={"extra": {"query": query[:80], "doc_count": len(docs), "latency_s": round(elapsed, 3)}})

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        content_preview = doc.page_content[:150].replace('\n', ' ')
        logger.debug("Document %d: source=%s content=%s...",
                      i, os.path.basename(source), content_preview,
                      extra={"extra": {"doc_index": i, "source": source}})

    # Format with source metadata
    retrieved_content = ""
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        retrieved_content += f"[Source: {source}]\n{doc.page_content}\n\n---\n\n"

    return retrieved_content.strip()

# ============================================================================
# GRAPH NODES
# ============================================================================

def get_user_context(state: State) -> dict:
    """Retrieve personalized context from Mem0"""
    user_id = state["mem0_user_id"]
    last_message = state["messages"][-1].content

    try:
        t0 = time.perf_counter()
        memories = mem0.search(last_message, filters={"user_id": user_id})
        elapsed = time.perf_counter() - t0
        memory_results = memories.get('results', [])

        if memory_results:
            context = "User's past interactions and preferences:\n"
            context += "\n".join([f"- {m['memory']}" for m in memory_results])
            logger.info("Found %d relevant memories for %s",
                         len(memory_results), user_id,
                         extra={"extra": {"user_id": user_id, "memory_count": len(memory_results), "latency_s": round(elapsed, 3)}})
        else:
            context = "No previous conversation history found."
            logger.info("No memories found for %s", user_id,
                         extra={"extra": {"user_id": user_id, "latency_s": round(elapsed, 3)}})

    except Exception as e:
        logger.warning("Mem0 search error", exc_info=True,
                        extra={"extra": {"user_id": user_id}})
        context = "Memory retrieval unavailable."

    return {"user_context": context}


def generate_query_or_respond(state: State) -> dict:
    """LLM decides whether to retrieve documents or respond directly"""
    user_context = state.get("user_context", "")

    system_prompt = f"""You are a helpful Linux virtualization testing assistant specializing in virttest APIs.

{user_context}

You have access to a virttest API knowledge base via the retrieve_documents tool. Use it when:
- The user asks about virttest functions, classes, or APIs
- The user needs specific API documentation or usage examples
- The question is about Linux virtualization testing tools
- You need to verify technical details

Respond directly WITHOUT the tool when:
- It's a simple greeting or acknowledgment
- The question is already fully answered in the user context/memory
- It's a conversational follow-up that doesn't need documentation

Answer only what is asked - be direct and focused."""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    t0 = time.perf_counter()
    response = llm.bind_tools([retrieve_documents]).invoke(messages)
    elapsed = time.perf_counter() - t0

    decision = "retrieve" if (hasattr(response, 'tool_calls') and response.tool_calls) else "respond_directly"
    logger.info("Agent routing decision: %s", decision,
                 extra={"extra": {"decision": decision, "latency_s": round(elapsed, 3)}})

    return {"messages": [response]}


def grade_documents(state: State) -> Literal["generate_answer", "rewrite_question"]:
    """Evaluate whether retrieved documents are relevant to the question"""
    question = next(
        m.content for m in reversed(state["messages"])
        if isinstance(m, HumanMessage)
    )
    retry_count = state.get("retry_count", 0)

    # Max retries to prevent infinite loops
    MAX_RETRIES = 2

    if retry_count >= MAX_RETRIES:
        logger.info("Max retries (%d) reached, proceeding with available docs",
                      MAX_RETRIES, extra={"extra": {"retry_count": retry_count}})
        return "generate_answer"

    # Find the tool response in messages
    tool_message = None
    for msg in reversed(state["messages"]):
        if hasattr(msg, 'content') and '---' in str(msg.content):
            tool_message = msg
            break

    if not tool_message:
        logger.info("No documents to grade, proceeding to answer")
        return "generate_answer"

    context = tool_message.content[:500]

    grading_prompt = f"""Grade the relevance of the retrieved context to the user's question.

Question: {question}

Retrieved Context:
{context}

Are these documents relevant to answering the question? Answer 'yes' or 'no'."""

    grader = llm.with_structured_output(GradeDocuments)
    t0 = time.perf_counter()
    result = grader.invoke([{"role": "user", "content": grading_prompt}])
    elapsed = time.perf_counter() - t0

    grade = result.binary_score
    if grade == "yes":
        logger.info("Documents graded: relevant",
                      extra={"extra": {"grade": grade, "retry_count": retry_count, "latency_s": round(elapsed, 3)}})
        return "generate_answer"
    else:
        logger.info("Documents graded: not relevant, rewriting (retry %d/%d)",
                      retry_count + 1, MAX_RETRIES,
                      extra={"extra": {"grade": grade, "retry_count": retry_count + 1, "latency_s": round(elapsed, 3)}})
        return "rewrite_question"


def rewrite_question(state: State) -> dict:
    """Reformulate the question for better retrieval"""
    original_question = next(
        m.content for m in reversed(state["messages"])
        if isinstance(m, HumanMessage)
    )
    retry_count = state.get("retry_count", 0)

    rewrite_prompt = f"""Rewrite the following question to be more specific and better suited for Linux documentation retrieval.
Focus on key technical terms and be concise.

Original question: {original_question}

Rewritten question:"""

    t0 = time.perf_counter()
    response = llm.invoke([{"role": "user", "content": rewrite_prompt}])
    elapsed = time.perf_counter() - t0
    rewritten = response.content

    logger.info("Query rewritten: '%s'", rewritten,
                 extra={"extra": {"original": original_question[:80], "rewritten": rewritten[:80], "latency_s": round(elapsed, 3)}})
    return {
        "messages": [HumanMessage(content=rewritten)],
        "retry_count": retry_count + 1
    }


def generate_answer(state: State) -> dict:
    """Generate final response using all available context"""
    user_context = state.get("user_context", "")

    system_prompt = f"""You are a helpful Linux virtualization testing assistant specializing in virttest APIs.

{user_context}

INSTRUCTIONS:
- List ALL relevant functions/APIs found in the retrieved documentation, not just one
- Use the retrieved documentation to provide accurate, specific details
- Be concise and direct
- If the documentation doesn't contain the answer, say "I don't have information about this in my documentation"
- Do not make assumptions or add information not in the retrieved documents"""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    t0 = time.perf_counter()
    response = llm.invoke(messages)
    elapsed = time.perf_counter() - t0

    logger.info("Answer generated",
                 extra={"extra": {"response_length": len(response.content), "latency_s": round(elapsed, 3)}})

    # Store the interaction in Mem0
    user_id = state["mem0_user_id"]
    try:
        current_question = next(
            m.content for m in reversed(state["messages"])
            if isinstance(m, HumanMessage)
        )
        interaction = [
            {"role": "user", "content": current_question},
            {"role": "assistant", "content": response.content}
        ]
        mem0.add(interaction, user_id=user_id)
        logger.info("Interaction stored in Mem0 for %s", user_id,
                      extra={"extra": {"user_id": user_id}})
    except Exception as e:
        logger.warning("Failed to store in Mem0", exc_info=True,
                        extra={"extra": {"user_id": user_id}})

    return {"messages": [response]}

# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_graph():
    """Construct the agentic RAG workflow"""
    workflow = StateGraph(State)

    # Add nodes (grade_documents is NOT a node, it's only a routing function)
    workflow.add_node("get_user_context", get_user_context)
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retrieve_documents]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)

    # Define flow
    workflow.add_edge(START, "get_user_context")
    workflow.add_edge("get_user_context", "generate_query_or_respond")

    # Conditional: retrieve documents or respond directly
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", END: "generate_answer"}
    )

    # After retrieval, grade documents and route accordingly
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question"
        }
    )

    # After rewrite, try again
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    # Final answer goes to END
    workflow.add_edge("generate_answer", END)

    return workflow.compile()

compiled_graph = build_graph()

# ============================================================================
# MAIN INTERFACE
# ============================================================================

def chat(user_input: str, user_id: str = "user1") -> str:
    """Run a single conversation turn"""
    t0 = time.perf_counter()
    logger.info("Request started",
                 extra={"extra": {"user_id": user_id, "input_length": len(user_input)}})

    state = {
        "messages": [HumanMessage(content=user_input)],
        "mem0_user_id": user_id,
        "retry_count": 0
    }

    graph_config = {"callbacks": langfuse_callbacks} if langfuse_callbacks else {}

    for event in compiled_graph.stream(state, config=graph_config):
        for node_name, node_output in event.items():
            if node_name != "__end__":
                logger.info("Node executed: %s", node_name)

    final_state = compiled_graph.invoke(state, config=graph_config)
    response = final_state["messages"][-1].content
    elapsed = time.perf_counter() - t0

    logger.info("Request completed",
                 extra={"extra": {"user_id": user_id, "total_latency_s": round(elapsed, 3), "response_length": len(response)}})

    return response


if __name__ == "__main__":
    try:
        compiled_graph.get_graph().draw_png("agentic_bookshelf_graph.png")
        logger.info("Graph visualization saved to agentic_bookshelf_graph.png")
    except Exception as e:
        logger.warning("Could not save graph visualization: %s", e)

    logger.info("Agentic Linux Knowledge Assistant started")
    print("\nType 'quit', 'exit', or 'bye' to end")
    print("Type '/memory' to view conversation context\n")

    user_id = "user1"

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'bye']:
                logger.info("User exited")
                break

            if user_input == '/memory':
                memories = mem0.search("", filters={"user_id": user_id})
                memory_results = memories.get('results', [])
                if memory_results:
                    print("\nYour conversation memory:")
                    for m in memory_results:
                        print(f"  - {m['memory']}")
                else:
                    print("\nNo memory found.")
                print()
                continue

            response = chat(user_input, user_id)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            logger.info("User interrupted")
            break
        except Exception as e:
            logger.error("CLI error", exc_info=True)

    if langfuse_handler:
        langfuse_handler._langfuse_client.flush()
        logger.info("LangFuse traces flushed")
