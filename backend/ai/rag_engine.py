import os
import json
import concurrent.futures
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- CONFIGURATION ---
JSON_FILE_PATH = os.path.join(os.getcwd(), "backend", "data", "processed", "legal_data_final.json")
DB_PATH = os.path.join(os.getcwd(), "backend", "data", "processed")
INDEX_NAME = "faiss_index"

class RAGEngine:
    def __init__(self):
        # --- EMBEDDING MODEL SETUP (Moved Inside Class) ---
        hf_token = os.getenv("HF_TOKEN")
        model_kwargs = {'token': hf_token} if hf_token else {}

        print("🔌 Loading Embedding Model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs=model_kwargs
        )

        self.db = None
        self.load_index()

    def load_index(self):
        if os.path.exists(os.path.join(DB_PATH, f"{INDEX_NAME}.faiss")):
            try:
                # Fixed: Use self.embeddings instead of global 'embeddings'
                self.db = FAISS.load_local(
                    DB_PATH, self.embeddings, index_name=INDEX_NAME, allow_dangerous_deserialization=True
                )
                print("✅ AI Memory Loaded Successfully!")
            except Exception as e:
                print(f"⚠️ Error loading memory: {e}")
        else:
            print("ℹ️ No memory found. Waiting for build...")

    def process_single_entry(self, entry):
        """Extract text from JSON entry."""
        try:
            text_content = entry.get('text') or entry.get('content') or str(entry)
            title = entry.get('title') or entry.get('section') or entry.get('source') or "Legal Document"
            if text_content:
                return Document(page_content=text_content, metadata={"title": title})
        except Exception:
            return None
        return None

    def build_index_from_json(self):
        print(f"🏗️  Loading data from: {JSON_FILE_PATH}")
        if not os.path.exists(JSON_FILE_PATH):
            print(f"❌ Error: File not found.")
            return

        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"⚡ Found {len(data)} records. Extracting text...")

        documents = []
        # Use Threading to extract text fast
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(self.process_single_entry, data)
            for doc in results:
                if doc: documents.append(doc)

        print(f"   -> Successfully extracted {len(documents)} documents.")

        # --- PRECISION SETTING ---
        # chunk_size=1000 means high detail. We do NOT miss anything.
        print("✂️  Splitting text into high-precision chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        final_docs = text_splitter.split_documents(documents)
        
        total_chunks = len(final_docs)
        print(f"🧠 Total Workload: {total_chunks} chunks to learn.")

        # --- SAFE BATCH PROCESSING ---
        # We process 2000 items at a time.
        # This keeps your RAM safe and prevents freezing.
        batch_size = 2000 
        
        print(f"🚀 Starting Training (Batch Size: {batch_size})...")
        
        # Create first batch to initialize DB
        print(f"   [1/{(total_chunks//batch_size)+1}] Processing first {batch_size} chunks...")
        
        # Fixed: Use self.embeddings
        self.db = FAISS.from_documents(final_docs[:batch_size], self.embeddings)
        
        # Loop through the rest
        current = batch_size
        batch_num = 2
        
        while current < total_chunks:
            end = min(current + batch_size, total_chunks)
            print(f"   [{batch_num}/{(total_chunks//batch_size)+1}] Processing chunks {current} to {end}...")
            
            # Add new batch to existing memory
            # Fixed: Use self.embeddings
            new_batch_db = FAISS.from_documents(final_docs[current:end], self.embeddings)
            self.db.merge_from(new_batch_db)
            
            current += batch_size
            batch_num += 1

        print("💾 Saving final memory to disk...")
        self.db.save_local(DB_PATH, index_name=INDEX_NAME)
        print("✅ SUCCESS! 100% of data has been learned.")

    def search(self, query, k=3):
        if not self.db: return []
        results = self.db.similarity_search(query, k=k)
        return [{"text": doc.page_content, "title": doc.metadata.get('title', 'Unknown')} for doc in results]

# Initialize global instance
rag = RAGEngine()

if __name__ == "__main__":
    rag.build_index_from_json()