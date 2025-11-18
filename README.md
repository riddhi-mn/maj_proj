# Hybrid RAG Chatbot

A hybrid RAG (Retrieval-Augmented Generation) chatbot that combines Neo4j graph database and Weaviate vector database for intelligent question answering about medicinal plants, illnesses, symptoms, and treatments.

## Architecture

- **Neo4j**: Graph database for structured relationships (Plant → Illness → Symptom → PreparationMethod)
- **Weaviate**: Vector database for PDF document storage and semantic search
- **LangChain**: Orchestrates the RAG pipeline with graph-enhanced retrieval
- **FastAPI**: Backend API server
- **Web UI**: Simple HTML/CSS/JS frontend

## Features

- Graph-enhanced vector search: Queries Neo4j first for context, then enhances Weaviate search
- PDF ingestion: Processes and chunks PDF documents for vector storage
- Hybrid retrieval: Combines structured graph knowledge with unstructured PDF content
- Web interface: Clean, modern chat UI

## Setup

### Prerequisites

- Python 3.8+
- Neo4j Desktop (create your graph manually)
- Docker & Docker Compose (for Weaviate)
- OpenAI API key (for LLM)

### Installation

1. **Clone and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create `.env` file** (copy from `.env.example`):
   ```env
   # Neo4j Desktop Connection
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password_here

   # Weaviate Connection
   WEAVIATE_URL=http://localhost:8080

   # LLM API Keys
   OPENAI_API_KEY=your_openai_api_key_here

   # Configuration (optional)
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   TOP_K_WEAVIATE=5
   TOP_K_NEO4J=10
   LLM_MODEL=gpt-3.5-turbo
   ```

3. **Start Weaviate:**
   ```bash
   docker-compose up -d
   ```

4. **Create Neo4j Graph in Neo4j Desktop:**
   - Open Neo4j Desktop
   - Create a new database
   - Create nodes: `Plant`, `Illness`, `Symptom`, `PreparationMethod`
   - Create relationships: `TREATS`, `HAS_SYMPTOM`, `USES_PREPARATION`
   - Populate with your 30 plants data

5. **Add PDFs:**
   - Place PDF files in `data/pdfs/` directory

6. **Start the server:**
   ```bash
   uvicorn main:app --reload
   ```

7. **Open in browser:**
   - Navigate to `http://localhost:8000`

## Usage

### Web Interface

1. Open `http://localhost:8000` in your browser
2. Click "Ingest PDFs" to process PDFs in `data/pdfs/`
3. Start chatting! Ask questions like:
   - "What plant helps with headaches?"
   - "How do I prepare willow for migraine treatment?"
   - "What are the symptoms of migraines?"

### API Endpoints

- `POST /api/chat` - Send a chat message
- `POST /api/ingest` - Ingest PDFs from `data/pdfs/`
- `POST /api/clear` - Clear chat memory
- `GET /health` - Check system status

## Project Structure

```
maj_proj/
├── main.py                 # FastAPI backend server
├── docker-compose.yml      # Weaviate service
├── requirements.txt        # Dependencies
├── src/
│   ├── config.py          # Configuration
│   ├── neo4j_client.py    # Neo4j client (read-only)
│   ├── weaviate_client.py # Weaviate client
│   ├── pdf_processor.py   # PDF processing
│   ├── graph_enhancer.py  # Graph-enhanced retrieval
│   ├── chatbot.py         # Chatbot orchestration
│   └── schemas.py         # Schema reference
├── data/
│   └── pdfs/              # PDF storage
├── static/
│   ├── css/
│   │   └── style.css      # Web UI styles
│   └── js/
│       └── app.js         # Web UI JavaScript
└── templates/
    └── index.html         # Web UI HTML
```

## Neo4j Graph Schema

The application expects the following schema in Neo4j Desktop:

- **Nodes**: `Plant`, `Illness`, `Symptom`, `PreparationMethod`
- **Relationships**: 
  - `Plant -[TREATS]-> Illness`
  - `Illness -[HAS_SYMPTOM]-> Symptom`
  - `Plant -[USES_PREPARATION]-> PreparationMethod`

## How It Works

1. **User Query**: "What plant helps with headaches?"
2. **Neo4j Query**: Extracts entities and finds related plants/illnesses/symptoms
3. **Graph Context**: Builds context string from graph results
4. **Enhanced Query**: Combines original query with graph context
5. **Weaviate Search**: Searches PDFs with enhanced query
6. **LLM Generation**: Combines graph context + PDF chunks → Answer

## License

MIT

