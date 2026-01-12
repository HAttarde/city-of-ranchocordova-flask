"""
ChromaDB Vector Store for Rancho Cordova Chatbot
=================================================

Provides persistent vector storage to avoid recomputing embeddings on every startup.
Uses file hashing to detect when source data changes and needs re-embedding.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings


class VectorStore:
    """
    Persistent vector store using ChromaDB.
    
    Features:
    - Persists embeddings to disk
    - Detects when source files change (via hash comparison)
    - Only re-embeds when necessary
    """
    
    def __init__(self, persist_dir: Optional[str] = None, collection_name: str = "rancho_cordova"):
        """
        Initialize the vector store.
        
        Args:
            persist_dir: Directory to store ChromaDB data. Defaults to ./chroma_db
            collection_name: Name of the ChromaDB collection
        """
        if persist_dir is None:
            persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
        
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.hash_file = os.path.join(persist_dir, "source_hashes.json")
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = None
        self._embedder = None
    
    def set_embedder(self, embedder):
        """Set the sentence transformer embedder."""
        self._embedder = embedder
    
    def initialize(self) -> bool:
        """
        Initialize or load the collection.
        
        Returns:
            True if collection was loaded from disk, False if newly created
        """
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=None  # We provide our own embeddings
            )
            print(f"✅ Loaded existing ChromaDB collection: {self.collection.count()} chunks")
            return True
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"description": "Rancho Cordova knowledge base"}
            )
            print(f"📦 Created new ChromaDB collection")
            return False
    
    def needs_rebuild(self, source_files: List[str]) -> bool:
        """
        Check if embeddings need to be rebuilt based on source file changes.
        
        Args:
            source_files: List of file paths to check
            
        Returns:
            True if any source file has changed
        """
        current_hashes = self._compute_file_hashes(source_files)
        stored_hashes = self._load_stored_hashes()
        
        if stored_hashes is None:
            print("📝 No stored hashes found - rebuild needed")
            return True
        
        for file_path, current_hash in current_hashes.items():
            stored_hash = stored_hashes.get(file_path)
            if stored_hash != current_hash:
                print(f"📝 File changed: {os.path.basename(file_path)} - rebuild needed")
                return True
        
        # Check if any files were removed
        for file_path in stored_hashes:
            if file_path not in current_hashes:
                print(f"📝 File removed: {os.path.basename(file_path)} - rebuild needed")
                return True
        
        print("✅ All source files unchanged - using cached embeddings")
        return False
    
    def add_chunks(self, chunks: List[str], chunk_types: List[str] = None):
        """
        Add chunks to the vector store with embeddings.
        
        Args:
            chunks: List of text chunks to embed and store
            chunk_types: Optional list of chunk type labels (for metadata)
        """
        if not chunks:
            print("⚠️ No chunks to add")
            return
        
        if self._embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")
        
        print(f"🔄 Embedding {len(chunks)} chunks...")
        
        # Generate embeddings
        embeddings = self._embedder.encode(chunks, convert_to_numpy=True).tolist()
        
        # Prepare metadata
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_type = chunk_types[i] if chunk_types else self._detect_chunk_type(chunk)
            metadatas.append({
                "type": chunk_type,
                "length": len(chunk)
            })
        
        # Generate IDs
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Clear existing data and add new
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=None,
            metadata={"description": "Rancho Cordova knowledge base"}
        )
        
        # Add in batches (ChromaDB has limits)
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=chunks[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
        
        print(f"✅ Added {len(chunks)} chunks to ChromaDB")
    
    def query(self, query_text: str, k: int = 5) -> List[str]:
        """
        Query the vector store for similar chunks.
        
        Args:
            query_text: The query string
            k: Number of results to return
            
        Returns:
            List of most similar chunk texts
        """
        if self._embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")
        
        if self.collection is None or self.collection.count() == 0:
            print("⚠️ Collection is empty")
            return []
        
        # Generate query embedding
        query_embedding = self._embedder.encode([query_text], convert_to_numpy=True).tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, self.collection.count())
        )
        
        # Extract documents
        if results and results["documents"]:
            return results["documents"][0]
        
        return []
    
    def save_hashes(self, source_files: List[str]):
        """Save current file hashes to disk."""
        hashes = self._compute_file_hashes(source_files)
        os.makedirs(os.path.dirname(self.hash_file), exist_ok=True)
        with open(self.hash_file, "w") as f:
            json.dump(hashes, f, indent=2)
        print(f"💾 Saved source file hashes")
    
    def get_chunk_count(self) -> int:
        """Get the number of chunks in the collection."""
        if self.collection is None:
            return 0
        return self.collection.count()
    
    def _compute_file_hashes(self, file_paths: List[str]) -> dict:
        """Compute MD5 hashes for a list of files."""
        hashes = {}
        for file_path in file_paths:
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    hashes[file_path] = hashlib.md5(f.read()).hexdigest()
        return hashes
    
    def _load_stored_hashes(self) -> Optional[dict]:
        """Load previously stored file hashes."""
        if not os.path.exists(self.hash_file):
            return None
        try:
            with open(self.hash_file, "r") as f:
                return json.load(f)
        except Exception:
            return None
    
    def _detect_chunk_type(self, chunk: str) -> str:
        """Detect the type of chunk from its content."""
        if chunk.startswith("ENERGY_RECORD"):
            return "energy"
        elif chunk.startswith("CS_RECORD"):
            return "customer_service"
        elif chunk.startswith("DEPT_RECORD"):
            return "department"
        elif chunk.startswith("UTILITY_"):
            return "utility"
        elif chunk.startswith("TOU_"):
            return "tou_rates"
        elif chunk.startswith("REBATE_"):
            return "rebate"
        elif chunk.startswith("PDF_DOCUMENT"):
            return "pdf"
        else:
            return "general"


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the singleton vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
