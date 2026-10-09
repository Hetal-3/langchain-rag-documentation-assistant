import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import (
    init_chat_model,
)  # used to initialise a chat client to make llm request
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / "langchain-docs" / ".env"
load_dotenv(env_path)

# Read API key
api_key = os.environ["OPEN_ROUTER_API_KEY"]

# Initialize the embeddings model
embeddings = OpenAIEmbeddings(
    model="nvidia/nemotron-3-embed-1b:free",
    openai_api_key=os.environ["OPEN_ROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
    check_embedding_ctx_length=False,  # <-- key fix: sends raw text, skips tiktoken
)

# size of model embeddings must be equal to dimensions of vector store
# Intialize the vector store
vectorstore = PineconeVectorStore(
    index_name="langchain-docs-2026", embedding=embeddings
)
# Intialize chat model
model = init_chat_model(
    "nvidia/nemotron-3.5-lightning:free",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
)


# query-->user query
# content and artifact return type should return 2 values
@tool(response_format="content_and_artifact")
def retrieve_content(query: str):
    """Retrieve relevant documentation to help answer user queries about langChain"""

    print("\nRETRIEVE_CONTENT CALLED")
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    # Serialize documents for the model
    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source','Unknown')}\n\nContent: {doc.page_content}"
        )
        for doc in retrieved_docs
    )
    # return both serialized content and raw documents
    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation

    Args:
        query (str): The user's question

    Returns:
        Dictionary containing:
        answer: The generated answer
        context: List of retrieved documentation
    """
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation."
        "Use the tool to find relevant documentation."
        "Always cite the sources in your answers."
        "If you cannot find the answer in the retrieved documentations say so"
    )
    # Last Line is to avoid hallucination

    agent = create_agent(model, tools=[retrieve_content], system_prompt=system_prompt)

    # Build messages list
    messages = [{"role": "user", "content": query}]

    # Invoke the agent
    response = agent.invoke({"messages": messages})

    # Extract the answer from last AI message
    answer = response["messages"][-1].content

    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of doc object
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    return {
        "answer": answer,
        "context": context_docs,
    }


if __name__ == "__main__":
    result = run_llm(query="Ola!How are you?")
    print(result)
