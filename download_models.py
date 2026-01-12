from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Downloading Qwen2.5-0.5B-Instruct...")
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

print("Downloading MiniLM embeddings...")
SentenceTransformer("all-MiniLM-L6-v2")

print("Download complete.")
