import os
import json
import pandas as pd
import joblib
import requests

def create_embedding(text_list):
    # Filter out empty strings to avoid Ollama NaN errors
    cleaned_text = [t.strip() for t in text_list if t and len(t.strip()) > 0]
    if not cleaned_text: return None

    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": cleaned_text
    })
    return r.json().get("embeddings")

json_folder = "jsons"
all_chunks_data = []

# 1. Iterate through all files
for file_name in os.listdir(json_folder):
    if file_name.endswith(".json"):
        file_path = os.path.join(json_folder, file_name)
        print(f"Processing: {file_name}")

        with open(file_path, 'r') as f:
            data = json.load(f)
            chunks = data.get('chunks', [])

        # 2. Process in batches to avoid API timeout/errors
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_text = [c['text'] for c in batch]
            
            embeddings = create_embedding(batch_text)
            
            if embeddings:
                for j, chunk in enumerate(batch):
                    # Attach the embedding to the chunk metadata
                    chunk['embedding'] = embeddings[j]
                    all_chunks_data.append(chunk)

# 3. Create a DataFrame and save to Joblib
df = pd.DataFrame(all_chunks_data)
joblib.dump(df, 'embeddings.joblib')

print(f"Successfully saved {len(df)} chunks to embeddings.joblib")