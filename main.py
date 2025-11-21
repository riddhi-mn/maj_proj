"""FastAPI backend server for the hybrid RAG chatbot."""
import os
import tempfile
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
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
graph_enhancer: Optional[GraphEnhancer] = None

# Session-based chatbots (session_id -> chatbot instance)
chatbots: Dict[str, HybridRAGChatbot] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    graph_context: Optional[str]
    sources: List[dict]


class IngestResponse(BaseModel):
    message: str
    chunks_processed: int
    files_processed: int


def get_or_create_chatbot(session_id: str) -> Optional[HybridRAGChatbot]:
    """Get or create a chatbot instance for a session."""
    global graph_enhancer, chatbots
    
    if not graph_enhancer:
        return None
    
    # Return existing chatbot for this session
    if session_id in chatbots:
        return chatbots[session_id]
    
    # Create new chatbot for this session
    try:
        chatbot = HybridRAGChatbot(graph_enhancer)
        chatbots[session_id] = chatbot
        print(f"✓ Created new chatbot for session: {session_id[:8]}...")
        return chatbot
    except Exception as e:
        print(f"✗ Failed to create chatbot for session {session_id}: {e}")
        return None


@app.on_event("startup")
async def startup():
    """Initialize database connections."""
    global neo4j_client, weaviate_client, graph_enhancer
    
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
            print("✓ Graph enhancer initialized successfully")
            print("✓ Chatbots will be created per session on first request")
        except Exception as e:
            print(f"✗ Graph enhancer initialization failed: {e}")
            graph_enhancer = None
    else:
        print("\n✗ Graph enhancer not initialized - missing database connections")
        print("   Fix the connection errors above and restart the server\n")


@app.on_event("shutdown")
async def shutdown():
    """Close database connections and clean up sessions."""
    global neo4j_client, weaviate_client, chatbots
    
    # Clear all chatbot sessions
    chatbots.clear()
    print("✓ Cleared all chatbot sessions")
    
    if neo4j_client:
        neo4j_client.close()
    if weaviate_client:
        weaviate_client.close()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests with session-based memory."""
    if not graph_enhancer:
        raise HTTPException(status_code=503, detail="Database connections not initialized. Check database connections.")
    
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    # Get or create chatbot for this session
    chatbot = get_or_create_chatbot(session_id)
    if not chatbot:
        raise HTTPException(status_code=503, detail="Failed to initialize chatbot for session.")
    
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
async def ingest_pdfs(files: List[UploadFile] = File(...)):
    """Ingest uploaded PDF files."""
    print(f"\n[Ingest] Received upload request with {len(files) if files else 0} files")
    
    if not weaviate_client:
        print("[Ingest] ✗ Weaviate not connected")
        raise HTTPException(status_code=503, detail="Weaviate not connected.")
    
    if not files or len(files) == 0:
        print("[Ingest] ✗ No files received")
        raise HTTPException(status_code=400, detail="No files uploaded. Please select at least one PDF file.")
    
    # Log file details
    for i, file in enumerate(files, 1):
        print(f"[Ingest] File {i}: {file.filename} (size: {file.size if hasattr(file, 'size') else 'unknown'})")
    
    # Validate all files are PDFs
    invalid_files = [f.filename for f in files if f.filename and not f.filename.lower().endswith('.pdf')]
    if invalid_files:
        raise HTTPException(
            status_code=400, 
            detail=f"Only PDF files are allowed. Invalid files: {', '.join(invalid_files)}"
        )
    
    processor = PDFProcessor()
    all_chunks = []
    processed_files = 0
    temp_files = []
    
    try:
        # Process each uploaded file
        for file in files:
            if not file.filename:
                continue
                
            # Save uploaded file temporarily
            temp_path = None
            try:
                # Read file content
                content = await file.read()
                
                if not content:
                    print(f"Warning: File {file.filename} is empty")
                    continue
                
                # Create temp file and write content
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb') as temp_file:
                    temp_path = temp_file.name
                    temp_files.append(temp_path)
                    temp_file.write(content)
                    temp_file.flush()
                
                print(f"✓ Saved temporary file: {temp_path}")
            except Exception as e:
                print(f"✗ Error saving file {file.filename}: {str(e)}")
                continue
            
            # Process the PDF
            try:
                print(f"Processing uploaded file: {file.filename}")
                if not temp_path or not os.path.exists(temp_path):
                    print(f"✗ Temp file not found for {file.filename}")
                    continue
                
                chunks = processor.extract_text_from_pdf(temp_path)
                
                # Update filename to use original filename instead of temp filename
                original_filename = file.filename
                for chunk in chunks:
                    chunk["filename"] = original_filename
                
                all_chunks.extend(chunks)
                processed_files += 1
                print(f"  Extracted {len(chunks)} chunks from {file.filename}")
            except Exception as e:
                print(f"Error processing {file.filename}: {str(e)}")
                # Continue with other files even if one fails
                continue
        
        # Add all chunks to Weaviate
        if all_chunks:
            weaviate_client.add_documents(all_chunks)
            return IngestResponse(
                message="PDFs ingested successfully",
                chunks_processed=len(all_chunks),
                files_processed=processed_files
            )
        else:
            return IngestResponse(
                message="No chunks extracted from uploaded files",
                chunks_processed=0,
                files_processed=processed_files
            )
            
    except Exception as e:
        import traceback
        error_detail = f"Error ingesting PDFs: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # Print full traceback to console
        raise HTTPException(status_code=500, detail=f"Error ingesting PDFs: {str(e)}")
    finally:
        # Clean up temporary files
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                print(f"Warning: Could not delete temp file {temp_path}: {e}")


class ClearRequest(BaseModel):
    session_id: Optional[str] = None


@app.post("/api/clear")
async def clear_memory(request: ClearRequest = ClearRequest()):
    """Clear chatbot conversation memory for a session."""
    if not graph_enhancer:
        raise HTTPException(status_code=503, detail="Database connections not initialized.")
    
    # If no session ID provided, clear all sessions
    if not request.session_id:
        chatbots.clear()
        return {"message": "All session memories cleared"}
    
    # Clear specific session
    if request.session_id in chatbots:
        chatbots[request.session_id].clear_memory()
        return {"message": f"Memory cleared for session: {request.session_id[:8]}..."}
    else:
        return {"message": "Session not found"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "neo4j": neo4j_client is not None,
        "weaviate": weaviate_client is not None,
        "graph_enhancer": graph_enhancer is not None,
        "active_sessions": len(chatbots)
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

