
from functools import lru_cache

from rag_answer import (
    PROJECT_ID,
    MAX_DISTANCE,
    get_openai_client,
    get_chroma_collection,
    generate_query_embedding,
    retrieve_context,
    prepare_context,
    generate_answer,
)


@lru_cache(maxsize=1)
def get_cached_openai_client():
    """Reuse one OpenAI client across requests."""
    return get_openai_client()


@lru_cache(maxsize=1)
def get_cached_chroma_collection():
    """Reuse the ChromaDB collection connection."""
    return get_chroma_collection()


def answer_question(question: str) -> dict:
    """
    Execute the existing Aqlithos RAG pipeline
    for a visitor's question.
    """

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    # 1. Initialize clients
    openai_client = get_cached_openai_client()
    collection = get_cached_chroma_collection()

    # 2. Generate query embedding
    query_vector = generate_query_embedding(
        openai_client,
        question,
    )

    # 3. Retrieve relevant documents
    retrieved_documents = retrieve_context(
        collection,
        query_vector,
    )

    if not retrieved_documents:
        return {
            "answer": (
                "The available Aqlithos information does not "
                "currently specify this. Please contact our "
                "team for confirmation."
            ),
            "project_id": PROJECT_ID,
            "sources": [],
        }

    # 4. Check retrieval relevance
    best_distance = retrieved_documents[0]["distance"]

    if best_distance > MAX_DISTANCE:
        return {
            "answer": (
                "The available Aqlithos information does not "
                "contain sufficiently relevant details to "
                "answer this question reliably. Please contact "
                "our team for confirmation."
            ),
            "project_id": PROJECT_ID,
            "sources": [],
        }

    # 5. Prepare retrieved context
    context = prepare_context(retrieved_documents)

    # 6. Generate the answer using the existing LLM logic
    answer = generate_answer(
        openai_client,
        question,
        context,
    )

    # 7. Prepare source information
    sources = []

    for index, item in enumerate(
        retrieved_documents,
        start=1,
    ):
        metadata = item["metadata"]

        sources.append({
            "source_number": index,
            "document_id": item["id"],
            "title": metadata.get("title", "Unknown"),
            "url": metadata.get("source_url", "Unknown"),
            "distance": round(item["distance"], 6),
        })

    return {
        "answer": answer,
        "project_id": PROJECT_ID,
        "sources": sources,
    }
