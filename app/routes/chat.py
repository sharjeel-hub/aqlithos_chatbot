
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.rag_service import answer_question


router = APIRouter(
    prefix="/api/v1",
    tags=["Chat"],
)


# ==========================================
# REQUEST MODEL
# ==========================================

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Visitor's question",
    )


# ==========================================
# SOURCE RESPONSE MODEL
# ==========================================

class SourceResponse(BaseModel):
    source_number: int
    document_id: str
    title: str
    url: str
    distance: float


# ==========================================
# CHAT RESPONSE MODEL
# ==========================================

class ChatResponse(BaseModel):
    answer: str
    project_id: str
    sources: list[SourceResponse]


# ==========================================
# CHAT ENDPOINT
# ==========================================

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask the Aqlithos AI assistant",
)
def chat(request: ChatRequest):
    """
    Receive a visitor's question and return
    an AI-generated answer with supporting sources.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        result = answer_question(question)
        return result

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception:
        # Keep internal errors and secrets out of API responses.
        raise HTTPException(
            status_code=500,
            detail="Unable to process your question right now.",
        )
