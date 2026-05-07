import os
from dotenv import load_dotenv

load_dotenv() 

HF_API_KEY = os.getenv("HF_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

print(f"HuggingFace Key Loaded: {bool(HF_API_KEY)}")
print(f"Groq Key Loaded: {bool(GROQ_API_KEY)}")

