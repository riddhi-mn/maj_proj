"""Weaviate client for vector database operations."""
from typing import List, Dict, Optional
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from src.config import Config


class WeaviateClient:
    """Client for Weaviate vector database operations."""
    
    COLLECTION_NAME = "Documents"
    
    def __init__(self):
        try:
            # Parse URL
            url = Config.WEAVIATE_URL.replace("http://", "").replace("https://", "")
            if ":" in url:
                host, port = url.split(":")
                port = int(port)
            else:
                host = url
                port = 8080
            
            # Connect with both HTTP and gRPC
            self.client = weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=False,
                grpc_host=host,
                grpc_port=50051,
                grpc_secure=False
            )
            
            # Test connection
            self.client.collections.list_all()
            self._ensure_collection()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Weaviate at {Config.WEAVIATE_URL}: {str(e)}. Make sure Weaviate is running with 'docker-compose up -d'")
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist, or recreate if it has wrong config."""
        collection_exists = self.client.collections.exists(self.COLLECTION_NAME)
        
        if collection_exists:
            # Try to delete existing collection to recreate with vectorizer
            # This ensures we have the correct vectorizer config
            try:
                print(f"⚠ Collection '{self.COLLECTION_NAME}' exists. Recreating with vectorizer...")
                self.client.collections.delete(self.COLLECTION_NAME)
            except Exception as e:
                print(f"⚠ Could not delete collection: {e}")
        
        # Create collection with vectorizer
        if not self.client.collections.exists(self.COLLECTION_NAME):
            self.client.collections.create(
                name=self.COLLECTION_NAME,
                vectorizer_config=Configure.Vectorizer.text2vec_openai(
                    model="ada",
                    model_version="002",
                    type_="text"
                ),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="filename", data_type=DataType.TEXT),
                    Property(name="page_number", data_type=DataType.INT),
                    Property(name="chunk_index", data_type=DataType.INT),
                ]
            )
            print(f"✓ Created collection '{self.COLLECTION_NAME}' with OpenAI vectorizer")
    
    def close(self):
        """Close the Weaviate connection."""
        if self.client:
            self.client.close()
    
    def add_documents(self, documents: List[Dict[str, any]]):
        """Add documents to Weaviate."""
        collection = self.client.collections.get(self.COLLECTION_NAME)
        
        with collection.batch.dynamic() as batch:
            for doc in documents:
                batch.add_object(
                    properties={
                        "content": doc["content"],
                        "filename": doc.get("filename", "unknown"),
                        "page_number": doc.get("page_number", 0),
                        "chunk_index": doc.get("chunk_index", 0),
                    },
                    vector=doc.get("vector")  # If you add embeddings, include here
                )
    
    def search(self, query: str, limit: int = None) -> List[Dict]:
        """Search documents by text (vector search)."""
        limit = limit or Config.TOP_K_WEAVIATE
        collection = self.client.collections.get(self.COLLECTION_NAME)
        
        # For now, using keyword search since we don't have vectorizer
        # In production, you'd want to embed the query and search by vector
        results = collection.query.bm25(
            query=query,
            limit=limit,
            return_metadata=["score"]
        )
        
        return [
            {
                "content": obj.properties["content"],
                "filename": obj.properties.get("filename", "unknown"),
                "page_number": obj.properties.get("page_number", 0),
                "score": obj.metadata.score if obj.metadata.score else 0.0
            }
            for obj in results.objects
        ]
    
    def hybrid_search(self, query: str, limit: int = None) -> List[Dict]:
        """Hybrid search combining keyword and vector (when available)."""
        limit = limit or Config.TOP_K_WEAVIATE
        collection = self.client.collections.get(self.COLLECTION_NAME)
        
        # BM25 search (keyword-based)
        results = collection.query.hybrid(
            query=query,
            alpha=0.7,  # 0.7 = more weight on keyword, 0.3 = more weight on vector
            limit=limit,
            return_metadata=["score"]
        )
        
        return [
            {
                "content": obj.properties["content"],
                "filename": obj.properties.get("filename", "unknown"),
                "page_number": obj.properties.get("page_number", 0),
                "score": obj.metadata.score if obj.metadata.score else 0.0
            }
            for obj in results.objects
        ]

