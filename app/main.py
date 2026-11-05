from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.chat import router as chat_router


app = FastAPI(
    title="Aqlithos AI Chatbot API",
    description=(
        "AI-powered chatbot API for Aqlithos "
        "using Retrieval-Augmented Generation (RAG)."
    ),
    version="1.0.0",
)

# Allow the local React development frontend to access this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(chat_router)


@app.get("/", tags=["General"])
def home():
    return {
        "message": "Welcome to Aqlithos AI Chatbot API",
        "project": "aqlithos",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "aqlithos-chatbot-api",
    }
