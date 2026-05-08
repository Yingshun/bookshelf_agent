"""
Agentic RAG with Mem0 - Complete Working Example
Combines personalized memory with intelligent document retrieval
Using DeepSeek for LLM and OpenAI for embeddings
"""

from typing import Annotated, TypedDict, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from mem0 import MemoryClient
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# DEEPSEEK CONFIGURATION
# ============================================================================

DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # or deepseek-reasoner

# ============================================================================
# DOCUMENTATION URLS - Load docs from these sources
# ============================================================================

DOCUMENTATION_URLS = [
    "https://docs.langchain.com/oss/python/langgraph/agentic-rag",
    "https://docs.mem0.ai/integrations/langgraph",
]

# Set to False to use SAMPLE_DOCUMENTS instead of loading from URLs
USE_LIVE_DOCS = os.getenv("USE_LIVE_DOCS", "true").lower() == "true"

# ============================================================================
# SAMPLE DATA - Replace with your actual documents
# ============================================================================

SAMPLE_DOCUMENTS = [
    """
    Product Documentation: CloudStore Pro

    CloudStore Pro is our enterprise cloud storage solution launched in 2024.

    Key Features:
    - 10TB storage per user
    - End-to-end encryption
    - Real-time collaboration
    - Version history (90 days)
    - API access with rate limits: 1000 requests/hour

    Pricing: $29/month per user
    """,
    """
    Product Documentation: CloudStore Pro - Advanced Features

    Advanced Features:
    - Custom integrations via REST API
    - Webhooks for file events
    - SAML SSO for enterprise
    - Audit logging and compliance reports
    - Dedicated support team

    API Authentication: Use Bearer tokens obtained from the developer portal.
    """,
    """
    Common Issues and Solutions:

    Q: Why can't I upload files larger than 5GB?
    A: The web interface has a 5GB limit. Use our desktop client or API for larger files.

    Q: How do I share files externally?
    A: Create a public link from the file menu. Links expire after 30 days by default.

    Q: What file formats are supported for preview?
    A: PDF, images (JPG, PNG), Office documents, and text files.
    """,
    """
    Security Best Practices:

    1. Enable two-factor authentication (2FA) for all users
    2. Use strong passwords with minimum 12 characters
    3. Regularly review access logs in the admin panel
    4. Set up IP allowlisting for sensitive data
    5. Enable automatic file scanning for malware

    Data is encrypted at rest using AES-256 and in transit using TLS 1.3.
    """
]

# ============================================================================
# CONFIGURATION
# ============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    mem0_user_id: str
    user_context: str

class GradeDocuments(BaseModel):
    """Schema for document relevance grading"""
    binary_score: Literal["yes", "no"] = Field(
        description="Relevance score: 'yes' if relevant, 'no' if not"
    )

# Initialize LLM (DeepSeek) and Mem0
llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=DEEPSEEK_API_BASE
)
mem0 = MemoryClient()

# ============================================================================
# DOCUMENT PROCESSING AND RETRIEVER SETUP
# ============================================================================

def setup_knowledge_base():
    """Initialize vector store with documents from URLs or sample documents"""

    documents = []

    if USE_LIVE_DOCS:
        print("Loading documents from URLs...")
        try:
            # Load documents from each URL
            for url in DOCUMENTATION_URLS:
                print(f"  → Fetching: {url}")
                loader = WebBaseLoader(url)
                docs = loader.load()
                documents.extend(docs)
                print(f"    ✓ Loaded {len(docs)} document(s)")

            print(f"✓ Successfully loaded {len(documents)} documents from {len(DOCUMENTATION_URLS)} URLs")

        except Exception as e:
            print(f"✗ Failed to load from URLs: {e}")
            print("  → Falling back to SAMPLE_DOCUMENTS")
            documents = [Document(page_content=doc.strip()) for doc in SAMPLE_DOCUMENTS]
    else:
        print("Using SAMPLE_DOCUMENTS (USE_LIVE_DOCS=false)")
        documents = [Document(page_content=doc.strip()) for doc in SAMPLE_DOCUMENTS]

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    doc_splits = text_splitter.split_documents(documents)

    # Create vector store
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings()
    )

    print(f"✓ Knowledge base initialized with {len(doc_splits)} chunks from {len(documents)} documents")
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = setup_knowledge_base()

# ============================================================================
# TOOL DEFINITION
# ============================================================================

@tool
def retrieve_documents(query: str) -> str:
    """Search the knowledge base for relevant product documentation and support information."""
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

    retrieved_content = "\n\n---\n\n".join([doc.page_content for doc in docs])
    return retrieved_content

# ============================================================================
# GRAPH NODES
# ============================================================================

def get_user_context(state: State) -> dict:
    """Retrieve personalized context from Mem0"""
    user_id = state["mem0_user_id"]
    last_message = state["messages"][-1].content

    try:
        memories = mem0.search(last_message, filters={"user_id": user_id}, limit=5)
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

    system_prompt = f"""You are a helpful customer support assistant for CloudStore Pro.

{user_context}

You have access to a knowledge base via the retrieve_documents tool. Use it when:
- The user asks about product features, pricing, or technical details
- You need factual information to answer accurately
- The question is about troubleshooting or how-to

Respond directly WITHOUT the tool when:
- The question is conversational or about their personal experience
- You can answer from the user context alone
- It's a simple greeting or acknowledgment
"""

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
        print("  ✗ Documents not relevant, rewriting question")
        return "rewrite_question"


def rewrite_question(state: State) -> dict:
    """Reformulate the question for better retrieval"""
    original_question = state["messages"][0].content

    rewrite_prompt = f"""Rewrite the following question to be more specific and better suited for document retrieval.
Focus on key terms and be concise.

Original question: {original_question}

Rewritten question:"""

    response = llm.invoke([{"role": "user", "content": rewrite_prompt}])
    rewritten = response.content

    print(f"  → Rewritten: '{rewritten}'")
    return {"messages": [HumanMessage(content=rewritten)]}


def generate_answer(state: State) -> dict:
    """Generate final response using all available context"""
    user_context = state.get("user_context", "")

    system_prompt = f"""You are a helpful customer support assistant for CloudStore Pro.

{user_context}

Provide accurate, helpful responses. If you have retrieved documentation, use it to give specific details.
Be concise but thorough. Personalize your response based on the user's history when relevant."""

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

    # Add nodes
    workflow.add_node("get_user_context", get_user_context)
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retrieve_documents]))
    workflow.add_node("grade_documents", grade_documents)
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

    # After retrieval, grade documents
    workflow.add_edge("retrieve", "grade_documents")

    # Conditional: use docs or rewrite question
    workflow.add_conditional_edges(
        "grade_documents",
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

graph = build_graph()

# ============================================================================
# MAIN INTERFACE
# ============================================================================

def chat(user_input: str, user_id: str = "default_user") -> str:
    """Run a single conversation turn"""
    print(f"\n{'='*60}")
    print(f"User ({user_id}): {user_input}")
    print(f"{'='*60}\n")

    state = {
        "messages": [HumanMessage(content=user_input)],
        "mem0_user_id": user_id
    }

    # Stream the graph execution
    result = None
    for event in graph.stream(state):
        for node_name, node_output in event.items():
            if node_name != "__end__":
                print(f"[{node_name}]")

    # Get final response
    final_state = graph.invoke(state)
    response = final_state["messages"][-1].content

    print(f"\n{'='*60}")
    print(f"Assistant: {response}")
    print(f"{'='*60}\n")

    return response


def interactive_mode():
    """Run interactive chat loop"""
    print("\n" + "="*60)
    print("🤖 Agentic RAG with Mem0 - Interactive Mode")
    print("="*60)
    print("\nType 'quit', 'exit', or 'bye' to end the conversation")
    print("Type 'switch <user_id>' to change user\n")

    user_id = "alice"
    print(f"Current user: {user_id}\n")

    while True:
        try:
            user_input = input(f"{user_id}> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye!\n")
                break

            if user_input.lower().startswith('switch '):
                user_id = user_input.split(' ', 1)[1].strip()
                print(f"\n✓ Switched to user: {user_id}\n")
                continue

            chat(user_input, user_id)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def run_demo():
    """Run a demo with sample queries"""
    print("\n" + "="*60)
    print("🎬 Running Demo Scenarios")
    print("="*60 + "\n")

    scenarios = [
        ("alice", "Hi! I'm interested in CloudStore Pro. What does it cost?"),
        ("alice", "What's the file size limit for uploads?"),
        ("bob", "How do I enable two-factor authentication?"),
        ("alice", "I need to share a large file with a client. What are my options?"),
        ("bob", "What encryption do you use?"),
    ]

    for user_id, query in scenarios:
        chat(query, user_id)
        input("\nPress Enter for next scenario...")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo()
    else:
        interactive_mode()
