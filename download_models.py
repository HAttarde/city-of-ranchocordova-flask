"""
Download Models Script
======================
Downloads only the sentence-transformer embeddings model.
The LLM is now served via Groq API (no local model needed).
"""

from sentence_transformers import SentenceTransformer

print("Downloading MiniLM embeddings model...")
SentenceTransformer("all-MiniLM-L6-v2")

print("✅ Download complete. (LLM is served via Groq API)")
