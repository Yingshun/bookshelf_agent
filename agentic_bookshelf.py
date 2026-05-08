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
from dotenv import load_dotenv

load_dotenv()

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
    docs = retriever.invoke(query)

    print(f"  → Retrieved {len(docs)} documents for query: '{query[:50]}...'")

    # Print detailed info for each retrieved document
    for i, doc in enumerate(docs, 1):
        print(f"\n     Document {i}:")
        # Print metadata
        for key, value in doc.metadata.items():
            if key == 'source':
                print(f"       {key}: {os.path.basename(value)}")
            elif isinstance(value, (list, dict)):
                print(f"       {key}: {str(value)[:100]}...")
            else:
                print(f"       {key}: {value}")
        # Print content preview
        content_preview = doc.page_content[:150].replace('\n', ' ')
        print(f"       content: {content_preview}...")

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
        memories = mem0.search(last_message, filters={"user_id": user_id})
        memory_results = memories.get('results', [])

        if memory_results:
            context = "User's past interactions and preferences:\n"
            context += "\n".join([f"- {m['memory']}" for m in memory_results])
            print(f"  → Found {len(memory_results)} relevant memories for {user_id}")
        else:
            context = "No previous conversation history found."
            print(f"  → No memories found for {user_id}")

    except Exception as e:
        print(f"  ⚠ Mem0 error: {e}")
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
    response = llm.bind_tools([retrieve_documents]).invoke(messages)

    if hasattr(response, 'tool_calls') and response.tool_calls:
        print("  → Agent decided to retrieve documents")
    else:
        print("  → Agent responding without retrieval")

    return {"messages": [response]}


def grade_documents(state: State) -> Literal["generate_answer", "rewrite_question"]:
    """Evaluate whether retrieved documents are relevant to the question"""
    question = state["messages"][0].content
    retry_count = state.get("retry_count", 0)

    # Max retries to prevent infinite loops
    MAX_RETRIES = 2

    if retry_count >= MAX_RETRIES:
        print(f"  → Max retries ({MAX_RETRIES}) reached, proceeding with available docs")
        return "generate_answer"

    # Find the tool response in messages
    tool_message = None
    for msg in reversed(state["messages"]):
        if hasattr(msg, 'content') and '---' in str(msg.content):
            tool_message = msg
            break

    if not tool_message:
        print("  → No documents to grade, proceeding to answer")
        return "generate_answer"

    context = tool_message.content[:500]  # First 500 chars for grading

    grading_prompt = f"""Grade the relevance of the retrieved context to the user's question.

Question: {question}

Retrieved Context:
{context}

Are these documents relevant to answering the question? Answer 'yes' or 'no'."""

    grader = llm.with_structured_output(GradeDocuments)
    result = grader.invoke([{"role": "user", "content": grading_prompt}])

    if result.binary_score == "yes":
        print("  ✓ Documents are relevant")
        return "generate_answer"
    else:
        print(f"  ✗ Documents not relevant, rewriting question (retry {retry_count + 1}/{MAX_RETRIES})")
        return "rewrite_question"


def rewrite_question(state: State) -> dict:
    """Reformulate the question for better retrieval"""
    original_question = state["messages"][0].content
    retry_count = state.get("retry_count", 0)

    rewrite_prompt = f"""Rewrite the following question to be more specific and better suited for Linux documentation retrieval.
Focus on key technical terms and be concise.

Original question: {original_question}

Rewritten question:"""

    response = llm.invoke([{"role": "user", "content": rewrite_prompt}])
    rewritten = response.content

    print(f"  → Rewritten: '{rewritten}'")
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
    response = llm.invoke(messages)

    # Store the interaction in Mem0
    user_id = state["mem0_user_id"]
    try:
        interaction = [
            {"role": "user", "content": state["messages"][0].content},
            {"role": "assistant", "content": response.content}
        ]
        mem0.add(interaction, user_id=user_id)
        print(f"  ✓ Interaction stored in Mem0 for {user_id}")
    except Exception as e:
        print(f"  ⚠ Failed to store in Mem0: {e}")

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
    print(f"\n{'='*60}")
    print(f"User ({user_id}): {user_input}")
    print(f"{'='*60}\n")

    state = {
        "messages": [HumanMessage(content=user_input)],
        "mem0_user_id": user_id,
        "retry_count": 0
    }

    # Stream the graph execution
    for event in compiled_graph.stream(state):
        for node_name, node_output in event.items():
            if node_name != "__end__":
                print(f"[{node_name}]")

    # Get final response
    final_state = compiled_graph.invoke(state)
    response = final_state["messages"][-1].content

    print(f"\n{'='*60}")
    print(f"Assistant: {response}")
    print(f"{'='*60}\n")

    return response


if __name__ == "__main__":
    # Draw graph visualization
    try:
        compiled_graph.get_graph().draw_png("agentic_bookshelf_graph.png")
        print("Graph visualization saved to agentic_bookshelf_graph.png\n")
    except Exception as e:
        print(f"Could not save graph visualization: {e}\n")

    print("\n" + "="*60)
    print("Agentic Linux Knowledge Assistant")
    print("="*60)
    print("\nType 'quit', 'exit', or 'bye' to end")
    print("Type '/memory' to view conversation context\n")

    user_id = "user1"

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nGoodbye!\n")
                break

            if user_input == '/memory':
                # Show memory context
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

            chat(user_input, user_id)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
