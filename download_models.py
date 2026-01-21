from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Downloading Llama Model...")
AutoTokenizer.from_pretrained("openlm-research/open_llama_13b")
AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_13b")

print("Downloading MiniLM embeddings...")
SentenceTransformer("all-MiniLM-L6-v2")

print("Download complete.")
