"""FastAPI backend server for the hybrid RAG chatbot."""
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from src.neo4j_client import Neo4jClient
from src.weaviate_client import WeaviateClient
from src.graph_enhancer import GraphEnhancer
from src.chatbot import HybridRAGChatbot
from src.pdf_processor import PDFProcessor

app = FastAPI(title="Hybrid RAG Chatbot")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients
neo4j_client: Optional[Neo4jClient] = None
weaviate_client: Optional[WeaviateClient] = None
chatbot: Optional[HybridRAGChatbot] = None


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    graph_context: Optional[str]
    sources: List[dict]


class IngestResponse(BaseModel):
    message: str
    chunks_processed: int


@app.on_event("startup")
async def startup():
    """Initialize database connections and chatbot."""
    global neo4j_client, weaviate_client, chatbot
    
    print("\n" + "="*50)
    print("Initializing database connections...")
    print("="*50)
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("\n⚠ WARNING: .env file not found!")
        print("Create a .env file with:")
        print("  NEO4J_URI=bolt://localhost:7687")
        print("  NEO4J_USER=neo4j")
        print("  NEO4J_PASSWORD=your_password")
        print("  WEAVIATE_URL=http://localhost:8080")
        print("  OPENAI_API_KEY=your_key\n")
    
    try:
        neo4j_client = Neo4jClient()
        print("✓ Connected to Neo4j")
    except Exception as e:
        print(f"✗ Neo4j connection failed: {e}")
        print("   → Make sure Neo4j Desktop is running")
        print("   → Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env")
        neo4j_client = None
    
    try:
        weaviate_client = WeaviateClient()
        print("✓ Connected to Weaviate")
    except Exception as e:
        print(f"✗ Weaviate connection failed: {e}")
        print("   → Run: docker-compose up -d")
        print("   → Check WEAVIATE_URL in .env (should be http://localhost:8080)")
        weaviate_client = None
    
    if neo4j_client and weaviate_client:
        try:
            graph_enhancer = GraphEnhancer(neo4j_client, weaviate_client)
            chatbot = HybridRAGChatbot(graph_enhancer)
            print("✓ Chatbot initialized successfully")
        except Exception as e:
            print(f"✗ Chatbot initialization failed: {e}")
            chatbot = None
    else:
        print("\n✗ Chatbot not initialized - missing database connections")
        print("   Fix the connection errors above and restart the server\n")


@app.on_event("shutdown")
async def shutdown():
    """Close database connections."""
    global neo4j_client, weaviate_client
    
    if neo4j_client:
        neo4j_client.close()
    if weaviate_client:
        weaviate_client.close()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests."""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized. Check database connections.")
    
    try:
        result = chatbot.query(request.message)
        return ChatResponse(
            response=result["response"],
            graph_context=result.get("graph_context"),
            sources=result.get("sources", [])
        )
    except Exception as e:
        import traceback
        error_detail = f"Error processing query: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # Print full traceback to console
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_pdfs():
    """Ingest PDFs from data/pdfs/ directory."""
    if not weaviate_client:
        raise HTTPException(status_code=503, detail="Weaviate not connected.")
    
    try:
        processor = PDFProcessor()
        chunks = processor.process_directory()
        
        if chunks:
            weaviate_client.add_documents(chunks)
            return IngestResponse(
                message="PDFs ingested successfully",
                chunks_processed=len(chunks)
            )
        else:
            return IngestResponse(
                message="No PDFs found to ingest",
                chunks_processed=0
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting PDFs: {str(e)}")


@app.post("/api/clear")
async def clear_memory():
    """Clear chatbot conversation memory."""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not initialized.")
    
    chatbot.clear_memory()
    return {"message": "Memory cleared"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "neo4j": neo4j_client is not None,
        "weaviate": weaviate_client is not None,
        "chatbot": chatbot is not None
    }


# Serve static files and frontend
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

if os.path.exists("templates"):
    @app.get("/")
    async def read_root():
        return FileResponse("templates/index.html")
else:
    @app.get("/")
    async def read_root():
        return {"message": "Hybrid RAG Chatbot API", "docs": "/docs"}

