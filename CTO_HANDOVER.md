# AQLITHOS AI CHATBOT
## Technical Project Handover for CTO

**Project:** Aqlithos AI Chatbot
**Project ID:** aqlithos
**Repository:** https://github.com/sharjeel-hub/aqlithos_chatbot
**Repository visibility:** Private
**Status:** Local development completed; production deployment pending.

---

# 1. Project Overview

The Aqlithos AI Chatbot is an AI-powered website assistant designed to answer visitors' questions about Aqlithos.

It uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the Aqlithos knowledge base and generate answers using an OpenAI language model.

The current implementation is designed specifically for Aqlithos. A generalized, multi-tenant Chatbot Builder platform is a possible future phase, not part of the current implementation.

# 2. Technology Stack

## Backend
- Python 3.13
- FastAPI 0.162.0
- Uvicorn 0.54.0
- Pydantic 2.14.2
- OpenAI Python SDK 3.41.0
- python-dotenv 1.2.4

## AI and RAG
- OpenAI embedding model: text-embedding-3-small
- Embedding dimensions: 1536
- Chat model: gpt-4.1-mini
- Retrieval-Augmented Generation (RAG)
- Top-K retrieval: 5
- Retrieval relevance threshold: 0.55

## Vector Database
- ChromaDB 1.5.9
- Docker container: aqlithos-chroma
- Collection: aqlithos_knowledge
- Project ID: aqlithos
- Current stored vectors: 86
- Docker named volume: aqlithos_chroma_data
- Container data directory: /data
- Local port: 8000

## Frontend
- React 19
- TypeScript 6
- Vite 8
- Tailwind CSS 4
- Lucide React

## Data Processing
- Python web scraping
- BeautifulSoup
- Requests
- lxml
- PyMuPDF
- Custom document chunking
- OpenAI embedding generation

# 3. Project Architecture

The current RAG workflow:

1. Collect website information using the scraping pipeline.
2. Clean and prepare collected documents.
3. Split documents into chunks.
4. Generate document embeddings.
5. Store vectors and metadata in ChromaDB.
6. Receive a visitor's question through the React frontend.
7. Send the question to FastAPI.
8. Generate an embedding for the new question.
9. Retrieve the five most relevant documents from ChromaDB.
10. Apply the retrieval relevance threshold.
11. Send relevant context to the OpenAI language model.
12. Return the answer and supporting source URLs to the frontend.

Existing document embeddings are stored in ChromaDB. New visitor questions are embedded individually for retrieval; existing document embeddings are not regenerated for every question.

# 4. Backend API

The FastAPI application is located in:

app/main.py

Chat route:

app/routes/chat.py

RAG service:

app/services/rag_service.py

Existing RAG logic:

rag_answer.py

## Available endpoints

### GET /
Returns API welcome information.

### GET /health
Returns backend health status.

### POST /api/v1/chat
Accepts a visitor question.

Request:

{
  "question": "What services does Aqlithos provide?"
}

Response structure:

{
  "answer": "Generated response",
  "project_id": "aqlithos",
  "sources": [
    {
      "source_number": 1,
      "document_id": "document-id",
      "title": "Source title",
      "url": "https://aqlithos.com/",
      "distance": 0.35
    }
  ]
}

# 5. Frontend

Frontend directory:

frontend/

Main files:

- src/App.tsx
- src/main.tsx
- src/index.css
- src/App.css
- src/assets/aqlithos-logo.png
- package.json
- package-lock.json
- vite.config.ts

The frontend currently provides:

- Aqlithos-branded chatbot interface.
- Responsive layout.
- Suggested questions.
- Chat message interface.
- Loading animation.
- Source links.
- New conversation action.
- FastAPI integration.

The frontend API base URL currently uses:

VITE_API_BASE_URL

Its local fallback is:

http://127.0.0.1:8001

Production API configuration is still required.

# 6. ChromaDB and Docker Database

The project uses ChromaDB, not MongoDB.

Existing local Docker configuration:

- Image: chromadb/chroma:1.5.9
- Container: aqlithos-chroma
- Named volume: aqlithos_chroma_data
- Container storage: /data
- Local port mapping: 127.0.0.1:8000 -> 8000

Collection:

aqlithos_knowledge

Current verified vector count:

86

The Docker volume stores the persistent database independently from the source code.

## Database backup

A database backup has been created outside the GitHub repository.

Filename:

aqlithos_chroma_backup.tar.gz

Local location:

Desktop/Aqlithos_Chatbot_Handover/

Archive size:

848,972 bytes

This archive must be transferred separately to the CTO. It is not included in GitHub.

## Database restoration outline

On the destination Docker host:

1. Install Docker.
2. Create the named volume:

docker volume create aqlithos_chroma_data

3. Copy the backup archive to the server.
4. Extract the archive into the named volume using a temporary Alpine container.
5. Start ChromaDB using the same image version, chromadb/chroma:1.5.9.
6. Verify collection aqlithos_knowledge.
7. Verify the stored vector count is 86.

The production deployment should keep ChromaDB on an internal Docker network and must not expose its database port publicly.

# 7. Knowledge Base and Source Files

Important files:

- aqlithos_documents_cleaned.jsonl
- scraper.py
- chunking.py
- embeddings.py
- chroma_store.py
- import_embeddings_to_chroma.py
- rag_answer.py
- test_retrieval.py

The cleaned knowledge-base JSONL file is approximately 127 KB.

The JSONL dataset and the ChromaDB backup are separate deliverables:

- JSONL contains cleaned source-document data.
- ChromaDB backup preserves the existing vector database.

Both are important for a complete handover.

# 8. Environment Configuration

The environment template is:

.env.example

Required variables:

OPENAI_API_KEY=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
CHROMA_HOST=127.0.0.1
CHROMA_PORT=8000

For Docker-based deployment, the backend should receive:

CHROMA_HOST=chroma
CHROMA_PORT=8000

The actual OPENAI_API_KEY must be supplied securely by the project owner.

The real .env file must never be committed to GitHub.

# 9. Local Development

## Start ChromaDB

docker start aqlithos-chroma

## Start backend

Activate the Python virtual environment, then run:

python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload

Local backend:

http://127.0.0.1:8001

API documentation:

http://127.0.0.1:8001/docs

## Start frontend

Open another terminal:

cd frontend
npm run dev

Local frontend:

http://localhost:5173

# 10. Completed and Verified Work

The following work has been completed and tested locally:

- Python backend structure established.
- FastAPI application created.
- Chat API endpoint implemented.
- RAG service integrated with FastAPI.
- OpenAI client integration implemented.
- ChromaDB Docker container configured.
- Persistent Docker volume configured.
- Aqlithos knowledge collection created.
- 86 stored vectors verified.
- Document retrieval tested.
- RAG answers and source references tested.
- React and TypeScript frontend created.
- Tailwind CSS configured.
- Aqlithos logo integrated.
- Chat interface and suggested questions implemented.
- Frontend production build successfully tested.
- FastAPI health endpoint tested.
- Local frontend-to-backend CORS configured.
- GitHub repository created and source code pushed.
- ChromaDB backup archive created.
- ChromaDB restored to running local state and its vector count verified.
- Environment template created.
- ChromaDB host and port made configurable.

# 11. Remaining Work — CTO Responsibilities

## A. Production Infrastructure
- Select and configure production hosting.
- Prepare Linux server and Docker environment.
- Create Dockerfile for the FastAPI backend.
- Create production Docker Compose configuration.
- Configure persistent ChromaDB storage.
- Restore and validate the ChromaDB backup.

## B. Backend Production Configuration
- Configure production environment variables.
- Configure production CORS origins.
- Replace local-only settings.
- Add API authentication or suitable access controls where required.
- Implement rate limiting and abuse protection.
- Add structured logging and error monitoring.
- Add reliable health checks and restart policies.

## C. Frontend Deployment
- Configure VITE_API_BASE_URL.
- Build the production frontend.
- Deploy the React static build.
- Verify all frontend API calls use HTTPS.

## D. Domain and Security
Target public domain:

chatbot.aqlithos.com

- Configure DNS.
- Configure HTTPS certificates.
- Secure API endpoint.
- Ensure ChromaDB is not publicly accessible.
- Review OpenAI API key handling.
- Configure backups and recovery procedures.

## E. Testing
- End-to-end frontend and backend testing.
- Verify retrieval quality.
- Verify source attribution.
- Test unavailable or irrelevant questions.
- Test multiple concurrent users.
- Test database restart and recovery.
- Test production performance.

## F. Website Integration
The current frontend is a full-page chatbot.

A floating chatbot widget that can be embedded into the main Aqlithos website is still pending.

## G. Future Phase
A multi-project Chatbot Builder platform may be developed later.

Potential future capabilities:
- Multiple client projects.
- Project-specific knowledge bases.
- Admin dashboard.
- Document upload and ingestion.
- Automated knowledge-base updates.
- Project-specific chatbot configuration.
- Usage and analytics dashboard.

This future platform is not part of the current Aqlithos-only MVP.

# 12. Important Handover Notes

GitHub repository:

https://github.com/sharjeel-hub/aqlithos_chatbot

Repository is private. CTO access must be granted by the repository owner.

The CTO must receive these separately:

1. GitHub repository access.
2. ChromaDB backup archive.
3. Securely supplied OpenAI API key.
4. This CTO handover document.

Never send API credentials through public GitHub commits.

---

**Prepared for:** Aqlithos (Private) Limited
**Project:** Aqlithos AI Chatbot
**Current milestone:** Local implementation and source-code handover
**Next milestone:** Production deployment and website integration
