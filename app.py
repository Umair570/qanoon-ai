import os
import sys
import json
import time
import threading
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv

from langchain_groq import ChatGroq

sys.path.append(os.getcwd()) 
load_dotenv()  

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("❌ ERROR: GROQ_API_KEY not found in .env file.")

try:
    llm = ChatGroq(
        temperature=0.2, # Low temperature keeps it analytical and factual
        model_name="llama-3.1-8b-instant", 
        api_key=groq_api_key
    )
    print("⚡ SUCCESS: Groq AI Model Ready!")
except Exception as e:
    print(f"❌ ERROR: Groq Initialization Failed - {e}")

print("🔌 Initializing Cloud Brain on Startup...")
rag = None
try:
    from backend.ai.rag_engine import RAGEngine
    rag = RAGEngine()
    
    print("🔥 Forcing Hugging Face API to wake up (This may take a minute)...")
    is_awake = False
    while not is_awake:
        try:
            rag.embeddings.embed_query("wake up")
            is_awake = True
            print("✅ SUCCESS: Hugging Face API is fully awake and ready!")
        except Exception:
            print("⏳ Hugging Face is still booting. Knocking again in 5 seconds...")
            time.sleep(5)
            
except Exception as e:
    print(f"❌ ERROR: Cloud AI Memory Failed - {e}")

def keep_brain_awake():
    while True:
        time.sleep(300) 
        if rag:
            try:
                rag.embeddings.embed_query("heartbeat ping")
                print("💓 [Heartbeat] Sent signal to keep Hugging Face awake.")
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
        "Qanoon AI has reached its maximum server capacity today. Please try again tomorrow!"
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
            # Reverted to k=5 to keep the context clean and prevent noise confusion
            docs = rag.search(user_text, k=5)
            if docs:
                context = ""
                for doc in docs:
                    context += f"\n--- SOURCE: {doc['title']} ---\n{doc['text']}\n"
        except Exception as e:
            def generic_error_message():
                yield f"<h3>⚠️ Memory Search Error</h3>An error occurred while searching the database: {str(e)}"
            return Response(stream_with_context(generic_error_message()), mimetype='text/plain')

    # --- THE STRICT RAG (ANTI-HALLUCINATION) PROMPT ---
    system_prompt = (
        "You are Qanoon AI, an expert legal advisor for Pakistani law.\n"
        "You MUST base your answer ENTIRELY on the provided DATA block below. You are strictly forbidden from using outside knowledge to guess an answer.\n\n"
        "🚨 STRICT RAG RULES:\n"
        "1. If the provided DATA contains information relevant to the user's query, answer it confidently. Do NOT use phrases like 'According to the data' or 'The provided text says'. Act like a legal expert stating facts.\n"
        "2. If the provided DATA does NOT contain the answer, you MUST NOT hallucinate or guess. You must respond EXACTLY with: '🛑 [REJECTED] I am sorry, but I do not have specific information regarding this in my current legal records.'\n"
        "3. If the query is abusive, offensive, or completely unrelated to law, respond EXACTLY with: '🛑 [REJECTED] I am Qanoon AI, a professional legal assistant. I can only answer questions related to Pakistani law.'\n\n"
        "💬 FORMATTING:\n"
        "- Answer concisely in a natural, conversational tone (max 3-4 sentences).\n"
        "- Use short bullet points if there are multiple rules or penalties to list.\n"
        "- **Bold** the actual penalties, prison times, or fine amounts.\n"
        "- If a specific Section is mentioned in the DATA, end your response on a new line with: '📖 Reference: Section [Number]'.\n"
    )

    full_prompt = f"{system_prompt}\n\nDATA:\n{context}\n\nQUERY: {user_text}"

    return Response(stream_with_context(generate_groq_response(full_prompt)), mimetype='text/plain')

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
    except Exception as e:
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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))