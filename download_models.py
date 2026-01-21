from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Downloading Llama-3.1-8B-Instruct...")
AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

print("Downloading MiniLM embeddings...")
SentenceTransformer("all-MiniLM-L6-v2")

print("Download complete.")
