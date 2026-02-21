import os
import json
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

# 1. Load Environment
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

# --- CONFIGURATION ---
JSON_PATH = "backend/data/processed/legal_data_final.json"
INDEX_NAME = "qanoon-ai"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 10 
METADATA_LIMIT = 38000 # Safety buffer below 40,960 bytes

def migrate_to_pinecone():
    print("🧠 Initializing Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"📂 Loading data from {JSON_PATH}...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        legal_data = json.load(f)
    
    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)

    print(f"🚀 Migrating {len(legal_data)} chunks...")

    success_count = 0
    for i in range(0, len(legal_data), BATCH_SIZE):
        batch_data = legal_data[i : i + BATCH_SIZE]
        vectors_to_upsert = []

        for idx, item in enumerate(batch_data):
            chunk_id = f"chunk_{i + idx}"
            
            # --- METADATA SAFETY CHECK ---
            raw_text = item['text']
            # If text is too long for Pinecone metadata, truncate it
            if len(raw_text.encode('utf-8')) > METADATA_LIMIT:
                print(f"⚠️ Truncating massive chunk at index {i+idx}")
                safe_text = raw_text.encode('utf-8')[:METADATA_LIMIT].decode('utf-8', 'ignore')
            else:
                safe_text = raw_text

            vector_values = embeddings.embed_query(raw_text) # Embed the FULL text
            
            metadata = {
                "text": safe_text, # Store the SAFE version
                "title": item.get('title', 'Unknown'),
                "source": item.get('source', 'Official PDF'),
                "type": item.get('type', 'Act')
            }

            vectors_to_upsert.append({
                "id": chunk_id,
                "values": vector_values,
                "metadata": metadata
            })

        try:
            index.upsert(vectors=vectors_to_upsert)
            success_count += len(batch_data)
            print(f"✅ [{success_count}/{len(legal_data)}] Chunks secured in cloud.")
            time.sleep(0.1) 
            
        except Exception as e:
            print(f"❌ Critical Error at batch {i//BATCH_SIZE}: {e}")
            break

    print(f"\n🎉 DONE! {success_count} chunks are now live in Pinecone.")

if __name__ == "__main__":
    migrate_to_pinecone()