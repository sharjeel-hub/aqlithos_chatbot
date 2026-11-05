
import os
from dotenv import dotenv_values
from openai import OpenAI
import chromadb


# ==============================
# AQLITHOS RAG CONFIGURATION
# ==============================

PROJECT_ID = "aqlithos"
COLLECTION_NAME = "aqlithos_knowledge"

CHROMA_HOST = "127.0.0.1"
CHROMA_PORT = 8000

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

TOP_K = 5

QUESTION = "What are Aqlithos' core services and solutions?"


def get_openai_client():
    """Load the API key without exposing it in the terminal."""

    config = dotenv_values(".env")
    api_key = config.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Check your project's .env file."
        )

    return OpenAI(api_key=api_key)


def generate_query_embedding(openai_client, question):
    """Convert the visitor's question into a vector."""

    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
        dimensions=EMBEDDING_DIMENSIONS,
    )

    vector = response.data[0].embedding

    if len(vector) != EMBEDDING_DIMENSIONS:
        raise ValueError("Unexpected embedding dimensions.")

    print(f"Query embedding generated: {len(vector)} dimensions")

    return vector


def retrieve_relevant_documents(query_vector):
    """Search the Aqlithos knowledge base in ChromaDB."""

    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
    )

    collection = client.get_collection(name=COLLECTION_NAME)

    print(f"Knowledge base records: {collection.count()}")

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=TOP_K,
        where={"project_id": PROJECT_ID},
        include=["documents", "metadatas", "distances"],
    )

    return results


def main():
    print("\n========== AQLITHOS RAG RETRIEVAL ==========\n")

    print(f"Question: {QUESTION}\n")

    openai_client = get_openai_client()

    query_vector = generate_query_embedding(
        openai_client,
        QUESTION,
    )

    results = retrieve_relevant_documents(query_vector)

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not ids:
        print("No relevant documents found.")
        return

    print(f"\nTop {len(ids)} relevant documents:\n")

    for rank, (doc_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1,
    ):
        print(f"--- Result {rank} ---")
        print(f"Document ID: {doc_id}")
        print(f"Title: {metadata.get('title', 'N/A')}")
        print(f"Source: {metadata.get('source_url', 'N/A')}")
        print(f"Chunk: {metadata.get('chunk_index', 'N/A')}")
        print(f"Cosine distance: {distance:.6f}")
        print(f"Content:\n{document[:500]}")
        print()

    print("========== RETRIEVAL COMPLETED ==========")


if __name__ == "__main__":
    main()
