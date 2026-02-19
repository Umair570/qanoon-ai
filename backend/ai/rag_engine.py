import os
import json
import concurrent.futures
from tqdm import tqdm  # Progress bar
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables (Critical for getting HF_TOKEN locally)
load_dotenv()

# --- CONFIGURATION ---
JSON_FILE_PATH = os.path.join(os.getcwd(), "backend", "data", "processed", "legal_data_final.json")
DB_PATH = os.path.join(os.getcwd(), "backend", "data", "processed")
INDEX_NAME = "faiss_index"

class RAGEngine:
    def __init__(self):
        print("🔌 Initializing Lightweight Cloud Brain (Hugging Face API)...")
        
        # Grab the token from Render Environment or local .env
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("⚠️ WARNING: HF_TOKEN not found! Embeddings API will fail.")

        # API-BASED EMBEDDINGS (Saves ~300MB of RAM)
        self.embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=hf_token,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.db = None
        self.load_index()

    def load_index(self):
        """Loads the FAISS index from disk."""
        index_file = os.path.join(DB_PATH, f"{INDEX_NAME}.faiss")
        if os.path.exists(index_file):
            try:
                self.db = FAISS.load_local(
                    DB_PATH, self.embeddings, index_name=INDEX_NAME, allow_dangerous_deserialization=True
                )
                print("✅ AI Memory Loaded Successfully!")
            except Exception as e:
                print(f"⚠️ Error loading memory: {e}")
        else:
            print("ℹ️ No memory found. Ready to build.")

    def process_single_entry(self, entry):
        """Fast JSON extraction."""
        try:
            text_content = entry.get('text') or entry.get('content') or str(entry)
            title = entry.get('title') or entry.get('section') or "Legal Document"
            if text_content:
                return Document(page_content=text_content, metadata={"title": title})
        except Exception:
            return None
        return None

    def build_index_from_json(self):
        """Extreme speed training with batching and progress tracking."""
        print(f"🏗️  Loading data from: {JSON_FILE_PATH}")
        if not os.path.exists(JSON_FILE_PATH):
            print(f"❌ Error: File not found.")
            return

        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"⚡ Extracting {len(data)} records via Threads...")
        documents = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(self.process_single_entry, data))
            documents = [doc for doc in results if doc]

        # SPEED TRICK: Larger chunk_size = fewer total chunks = FASTER TRAINING
        print("✂️  Splitting into 1000-character chunks (Best for Speed/Accuracy balance)...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        final_docs = text_splitter.split_documents(documents)
        
        total_chunks = len(final_docs)
        print(f"🚀 Training on {total_chunks} chunks...")

        # Processing in batches with a Progress Bar
        batch_size = 5000 
        self.db = None

        for i in tqdm(range(0, total_chunks, batch_size), desc="Training Progress"):
            batch = final_docs[i : i + batch_size]
            if self.db is None:
                self.db = FAISS.from_documents(batch, self.embeddings)
            else:
                self.db.add_documents(batch)

        print("💾 Saving final memory to disk...")
        self.db.save_local(DB_PATH, index_name=INDEX_NAME)
        print("✅ SUCCESS! 100% of data has been learned.")

    def search(self, query, k=3):
        if not self.db:
            return []
        results = self.db.similarity_search(query, k=k)
        return [{"text": doc.page_content, "title": doc.metadata.get('title', 'Unknown')} for doc in results]

# --- THE WINDOWS SAFETY GUARD ---
if __name__ == "__main__":
    # Create the engine inside the main block to prevent infinite spawn loops on Windows
    rag = RAGEngine()
    rag.build_index_from_json()