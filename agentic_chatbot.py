"""
Gradio web UI for Agentic RAG chatbot
Wraps the agentic_bookshelf.py LangGraph pipeline
"""

import warnings
warnings.filterwarnings('ignore', message='.*Qdrant client version.*')
warnings.filterwarnings('ignore', message='.*Pydantic V1.*')

import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from agentic_bookshelf import compiled_graph
from mem0 import Memory
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Mem0 client (same config as agentic_bookshelf.py)
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

user_id = "user1"
last_memory = ""

def respond(message, history):
    """
    Handle user input and return response from agentic RAG system
    Gradio 6.x ChatInterface signature: fn(message: dict | str, history: list)
    """
    global last_memory

    # Extract text from message (Gradio 6.x can pass dict or string)
    if isinstance(message, dict):
        user_input = message.get('text', '')
    else:
        user_input = message

    # Special command: show memory
    if user_input.strip() == "/memory":
        if last_memory:
            return f"Memory context: {last_memory}"
        return "No memory context available."

    # Special command: show all memories
    if user_input.strip() == "/memories":
        try:
            memories = mem0.search("", filters={"user_id": user_id})
            memory_results = memories.get('results', [])
            if memory_results:
                output = "Your conversation memory:\n"
                for m in memory_results[:10]:  # Limit to 10
                    output += f"- {m['memory']}\n"
                return output
            else:
                return "No memories found."
        except Exception as e:
            return f"Error retrieving memories: {e}"

    MAX_HISTORY_TURNS = 10
    messages = []
    if history:
        recent = history[-MAX_HISTORY_TURNS:]
        for entry in recent:
            if isinstance(entry, dict):
                role = entry.get("role", "")
                content = entry.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=user_input))

    state = {
        "messages": messages,
        "mem0_user_id": user_id,
        "retry_count": 0
    }

    response_text = ""
    try:
        # Stream the graph execution
        for event in compiled_graph.stream(state):
            for value in event.values():
                if value.get("messages"):
                    response_text = value["messages"][-1].content
                if value.get("user_context"):
                    last_memory = value["user_context"]

        if not response_text:
            return "No response generated."

        return response_text

    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Agentic Linux Assistant") as demo:
    gr.Markdown("""
    # Agentic Linux Knowledge Assistant

    An intelligent RAG chatbot that:
    - Decides when to retrieve documentation
    - Grades document relevance
    - Rewrites queries for better results
    - Remembers your conversation history

    **Commands:**
    - `/memory` - Show memory context from last query
    - `/memories` - Show all your conversation memories
    """)

    chatbot = gr.ChatInterface(
        fn=respond,
        examples=[
            "How do I use VMXML.new_from_dumpxml()?",
            "What's the difference between hotplug and unplug?",
            "How can I wait for a condition with timeout?",
            "Show me how to create a Linux bridge"
        ]
    )

    gr.Markdown("""
    ---
    Tip: The assistant will intelligently decide whether to search documentation or answer from memory.
    """)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Agentic Linux Knowledge Assistant")
    print("="*60 + "\n")
    print("Server will be available at: http://localhost:7860")
    print("Press Ctrl+C to stop\n")

    demo.launch(server_name="0.0.0.0", share=False)
