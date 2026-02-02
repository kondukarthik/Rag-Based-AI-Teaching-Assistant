import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests
import sys

# --- CONFIGURATION ---
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "llama3.2"  # or "deepseek-r1"
OLLAMA_API = "http://localhost:11434/api"

def create_embedding(text_list):
    """Generates vectors for a list of strings using Ollama."""
    try:
        # Ensure we don't send empty strings which can crash some models
        text_list = [t for t in text_list if t.strip()]
        if not text_list:
            return None

        r = requests.post(f"{OLLAMA_API}/embed", json={
            "model": EMBEDDING_MODEL,
            "input": text_list
        })
        r.raise_for_status()
        return r.json().get("embeddings")
    except Exception as e:
        print(f"Embedding Error: {e}")
        return None

def inference(prompt):
    """Sends the final RAG prompt to the LLM."""
    try:
        r = requests.post(f"{OLLAMA_API}/generate", json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        })
        r.raise_for_status()
        response_json = r.json()
        
        # Check if Ollama returned an internal error (like exit status 2)
        if "error" in response_json:
            return {"error": response_json["error"]}
        
        return {"response": response_json.get("response", "No response text found.")}
    
    except requests.exceptions.ConnectionError:
        return {"error": "Ollama is not running. Please start the Ollama app."}
    except Exception as e:
        return {"error": str(e)}

# 1. Load the database
try:
    df = joblib.load('embeddings.joblib')
    print(f"--- Loaded {len(df)} video chunks ---")
except FileNotFoundError:
    print("Error: 'embeddings.joblib' not found. Run your preprocess script first!")
    sys.exit()

# 2. Get User Input
incoming_query = input("\nAsk a Question: ").strip()
if not incoming_query:
    print("Empty question. Exiting.")
    sys.exit()

# 3. Vector Search (Retrieval)
print("Searching for relevant video segments...")
question_embedding = create_embedding([incoming_query])

if question_embedding is None:
    print("Failed to create embedding for your question.")
    sys.exit()

# Calculate similarity
stored_embeddings = np.vstack(df['embedding'].values)
similarities = cosine_similarity(stored_embeddings, [question_embedding[0]]).flatten()

# Get Top 5 results
top_results = 5
max_indices = similarities.argsort()[::-1][:top_results]
context_df = df.iloc[max_indices]

# 4. Construct the Prompt (Augmentation)
context_json = context_df[["title", "number", "start", "end", "text"]].to_json(orient="records")

prompt = f'''
I am teaching web development in my Sigma web development course. 
Below are video subtitle chunks as context. Use them to answer the user's question.

CONTEXT FROM VIDEOS:
{context_json}

USER QUESTION:
"{incoming_query}"

INSTRUCTIONS:
1. Answer in a human-friendly way.
2. Tell the user exactly which video (title and number) and what timestamp (start time) contains the answer.
3. Guide them to go to that specific video.
4. If the question is not related to the provided context, politely state you only answer questions about the Sigma course.
5. Do not mention "JSON" or "chunks" in your answer.
'''

# 5. Generate Answer (Generation)
print("Generating AI response...")
result = inference(prompt)

if "error" in result:
    print(f"\n[!] AI Error: {result['error']}")
    print("Tip: If you see 'exit status 2', try restarting Ollama or closing heavy apps.")
else:
    answer = result["response"]
    print("\n--- AI RESPONSE ---")
    print(answer)
    
    # Save for reference
    with open("prompt.txt", "w", encoding="utf-8") as f: f.write(prompt)
    with open("response.txt", "w", encoding="utf-8") as f: f.write(answer)