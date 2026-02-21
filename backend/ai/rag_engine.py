import os
import json
import concurrent.futures
from tqdm import tqdm  # Progress bar
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore # UPDATED
from langchain_core.documents import Document
from dotenv import load_dotenv
from pinecone import Pinecone # UPDATED

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
JSON_FILE_PATH = os.path.join(os.getcwd(), "backend", "data", "processed", "legal_data_final.json")
INDEX_NAME = "qanoon-ai" # Must match your Pinecone Dashboard

class RAGEngine:
    def __init__(self):
        print("🔌 Initializing Cloud Brain (Hugging Face + Pinecone)...")
        
        hf_token = os.getenv("HF_TOKEN")
        pc_api_key = os.getenv("PINECONE_API_KEY") # NEW

        if not hf_token:
            print("⚠️ WARNING: HF_TOKEN not found!")
        if not pc_api_key:
            print("⚠️ WARNING: PINECONE_API_KEY not found!")

        # Embeddings API
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=hf_token
        )
        
        self.db = None
        self.load_index()

    def load_index(self):
        """Connects to the existing Pinecone index."""
        try:
            # UPDATED: Direct connection to Pinecone Cloud instead of local disk
            self.db = PineconeVectorStore(
                index_name=INDEX_NAME,
                embedding=self.embeddings,
                pinecone_api_key=os.getenv("PINECONE_API_KEY")
            )
            print("✅ Cloud AI Memory Connected Successfully!")
        except Exception as e:
            print(f"⚠️ Error connecting to cloud memory: {e}")

    def process_single_entry(self, entry):
        """Fast JSON extraction."""
        try:
            text_content = entry.get('text') or entry.get('content') or str(entry)
            title = entry.get('title') or entry.get('section') or "Legal Document"
            source = entry.get('source') or "Official PDF"
            doc_type = entry.get('type') or "Act"

            if text_content:
                # Optimized metadata for Pinecone
                return Document(
                    page_content=text_content, 
                    metadata={
                        "title": title,
                        "source": source,
                        "type": doc_type
                    }
                )
        except Exception:
            return None
        return None

    def build_index_from_json(self):
        """Extreme speed training with batching and cloud upsert."""
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

        print("✂️  Splitting into chunks (1000 size)...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        final_docs = text_splitter.split_documents(documents)
        
        total_chunks = len(final_docs)
        print(f"🚀 Pushing {total_chunks} chunks to Pinecone...")

        # UPDATED: Use smaller batch size for Cloud stability (avoid 2MB/4MB errors)
        batch_size = 50 
        
        # Initialize index
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        # Clear existing data if you want a fresh start (Optional)
        # pc.Index(INDEX_NAME).delete(delete_all=True)

        for i in tqdm(range(0, total_chunks, batch_size), desc="Cloud Upload Progress"):
            batch = final_docs[i : i + batch_size]
            
            # Metadata Safety: Truncate text if it somehow exceeds 40KB (rare for 1k chunks)
            for doc in batch:
                if len(doc.page_content.encode('utf-8')) > 38000:
                    doc.page_content = doc.page_content[:38000]

            if self.db is None:
                self.db = PineconeVectorStore.from_documents(
                    batch, self.embeddings, index_name=INDEX_NAME
                )
            else:
                self.db.add_documents(batch)

        print("✅ SUCCESS! 100% of data is now in the Pinecone Cloud.")

    def search(self, query, k=5): # Defaulted to 5 for better cloud retrieval
        if not self.db:
            print("❌ Search failed: Database not initialized.")
            return []
        try:
            results = self.db.similarity_search(query, k=k)
            return [
                {
                    "text": doc.page_content, 
                    "title": doc.metadata.get('title', 'Unknown')
                } for doc in results
            ]
        except Exception as e:
            print(f"⚠️ Search error: {e}")
            return []

# --- THE WINDOWS SAFETY GUARD ---
if __name__ == "__main__":
    rag = RAGEngine()
    # Only run this if you need to re-upload everything
    # rag.build_index_from_json()