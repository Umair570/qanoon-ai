import os
import sys
import json
import time
import threading
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv

from langchain_groq import ChatGroq

# Ensure local imports work correctly
sys.path.append(os.getcwd()) 
load_dotenv()  

# --- API KEYS & CONFIG ---
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = "qanoon-ai"

if not groq_api_key:
    print("❌ ERROR: GROQ_API_KEY not found in environment.")
if not pinecone_api_key:
    print("❌ ERROR: PINECONE_API_KEY not found in environment.")

# Initialize LLM
try:
    llm = ChatGroq(
        temperature=0.2, 
        model_name="llama-3.1-8b-instant", 
        api_key=groq_api_key
    )
    print("⚡ SUCCESS: Groq AI Model Ready!")
except Exception as e:
    print(f"❌ ERROR: Groq Initialization Failed - {e}")

print("🔌 Initializing Pinecone Cloud Brain on Startup...")
rag = None
try:
    # This calls your RAGEngine class
    from backend.ai.rag_engine import RAGEngine
    rag = RAGEngine()
    
    # Render Memory-Safe Wakeup: Ping the API instead of loading a local model
    print("🔥 Forcing Cloud Embedding API to wake up...")
    is_awake = False
    while not is_awake:
        try:
            rag.embeddings.embed_query("wake up")
            is_awake = True
            print("✅ SUCCESS: Cloud Memory is fully awake!")
        except Exception:
            print("⏳ Cloud API is still booting. Knocking again in 5 seconds...")
            time.sleep(5)
            
except Exception as e:
    print(f"❌ ERROR: Cloud AI Memory Failed - {e}")

# Keep-alive heartbeat (Critical for Hugging Face Inference API)
def keep_brain_awake():
    while True:
        time.sleep(300) 
        if rag:
            try:
                rag.embeddings.embed_query("heartbeat ping")
                print("💓 [Heartbeat] Sent signal to keep Cloud Brain awake.")
            except Exception:
                pass 

threading.Thread(target=keep_brain_awake, daemon=True).start()

app = Flask(__name__)

def generate_groq_response(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            for chunk in llm.stream(prompt):
                if chunk.content:
                    yield chunk.content
            return 
        except Exception as e:
            error_msg = str(e).lower()
            if '429' in error_msg or 'rate_limit' in error_msg:
                wait_time = (attempt + 1) * 5 
                time.sleep(wait_time)
            else:
                yield f"⚠️ API Error: {str(e)}"
                return

    yield (
        "<h3>⚠️ Daily Limit Reached</h3>"
        "Qanoon AI has reached its maximum server capacity. Please try again tomorrow!"
    )

@app.route('/')
def home(): return render_template('index.html')

@app.route('/consult', methods=['POST'])
def consult():
    data = request.json
    user_text = data.get('text', '')
    
    print(f"🔍 Analyzing: {user_text}")
    context = "No specific legal document found."
    
    if rag:
        try:
            # Query the cloud index
            docs = rag.search(user_text, k=5)
            if docs:
                context = ""
                for doc in docs:
                    title = doc.get('title', 'Unknown Source')
                    text = doc.get('text', 'No content available')
                    context += f"\n--- SOURCE: {title} ---\n{text}\n"
        except Exception as e:
            def generic_error_message():
                yield f"<h3>⚠️ Memory Search Error</h3>Cloud retrieval failed: {str(e)}"
            return Response(stream_with_context(generic_error_message()), mimetype='text/plain')

    # THE STRICT RAG (ANTI-HALLUCINATION) PROMPT
    system_prompt = (
        "You are Qanoon AI, an expert legal advisor for Pakistani law.\n"
        "You MUST base your answer ENTIRELY on the provided DATA block below. You are strictly forbidden from using outside knowledge to guess an answer.\n\n"
        "🚨 STRICT RAG RULES:\n"
        "1. If the provided DATA contains information relevant to the user's query, answer it confidently. Act like a legal expert stating facts.\n"
        "2. If the provided DATA does NOT contain the answer, respond EXACTLY with: '🛑 [REJECTED] I am sorry, but I do not have specific information regarding this in my current legal records.'\n"
        "3. If the query is abusive or unrelated to law, respond EXACTLY with: '🛑 [REJECTED] I am Qanoon AI, a professional legal assistant. I can only answer questions related to Pakistani law.'\n\n"
        "💬 FORMATTING:\n"
        "- Answer concisely in a natural, conversational tone (max 3-4 sentences).\n"
        "- Use short bullet points if there are multiple rules or penalties.\n"
        "- **Bold** the actual penalties, prison times, or fine amounts.\n"
        "- Always end your response on a new line with: '📖 Reference: [Document Title]' using the SOURCE provided in the DATA.\n"
    )

    full_prompt = f"{system_prompt}\n\nDATA:\n{context}\n\nQUERY: {user_text}"
    return Response(stream_with_context(generate_groq_response(full_prompt)), mimetype='text/plain')

# Lawyers database logic remains unchanged
LAWYERS_DB_PATH = os.path.join("backend", "data", "raw", "lawyers_db.json")

@app.route('/lawyers', methods=['GET'])
def get_lawyers():
    all_lawyers = []
    filtered_lawyers = []
    category = request.args.get('category', 'general').lower().strip()
    
    try:
        if os.path.exists(LAWYERS_DB_PATH):
            with open(LAWYERS_DB_PATH, 'r', encoding='utf-8') as f:
                all_lawyers = json.load(f)
        else:
            return jsonify([]) 
    except Exception:
        return jsonify([])

    if not all_lawyers:
        return jsonify([])

    if category == 'general' or not category:
        return jsonify(all_lawyers[:10])
    
    for lawyer in all_lawyers:
        lawyer_tags = [t.lower() for t in lawyer.get('tags', [])]
        lawyer_specialty = lawyer.get('specialty', '').lower()
        if category in lawyer_tags or category in lawyer_specialty:
            filtered_lawyers.append(lawyer)
    
    if not filtered_lawyers:
        return jsonify(all_lawyers[:5])
        
    return jsonify(filtered_lawyers)

if __name__ == "__main__":
    # Render deployment port binding
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))