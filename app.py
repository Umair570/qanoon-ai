import os
import sys
import json
import time
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv

# 1. IMPORT GROQ
from langchain_groq import ChatGroq

# Setup paths & security
sys.path.append(os.getcwd()) 
load_dotenv()  

# 2. CONFIGURE GROQ CLIENT
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("❌ ERROR: GROQ_API_KEY not found in .env file.")

try:
    llm = ChatGroq(
        temperature=0.1, 
        model_name="llama-3.1-8b-instant", 
        api_key=groq_api_key
    )
    print("⚡ SUCCESS: Groq AI Model Ready!")
except Exception as e:
    print(f"❌ ERROR: Groq Initialization Failed - {e}")

app = Flask(__name__)

# --- NEW: LAZY LOAD RAG (This fixes the Render Timeout!) ---
rag = None
is_brain_loaded = False

def get_rag():
    global rag, is_brain_loaded
    if not is_brain_loaded:
        try:
            print("🔌 Waking up Cloud Brain (This takes a moment)...")
            from backend.ai.rag_engine import RAGEngine
            rag = RAGEngine()
            is_brain_loaded = True
            print("✅ SUCCESS: Cloud AI Memory Loaded!")
        except Exception as e:
            print(f"❌ ERROR: Cloud AI Memory Failed - {e}")
    return rag

# --- STREAM GENERATOR WITH RATE LIMIT SAFETY ---
def generate_groq_response(prompt, max_retries=3):
    """Streams the response instantly, handling Daily/Minute Quota limits."""
    for attempt in range(max_retries):
        try:
            for chunk in llm.stream(prompt):
                if chunk.content:
                    yield chunk.content
            return # Exit successfully if stream finishes

        except Exception as e:
            error_msg = str(e).lower()
            
            if '429' in error_msg or 'rate_limit' in error_msg:
                wait_time = (attempt + 1) * 5 
                print(f"⚠️ API Minute Limit hit. Sleeping for {wait_time}s...")
                time.sleep(wait_time)
            else:
                yield f"⚠️ API Error: {str(e)}"
                return

    yield (
        "<h3>⚠️ Daily Limit Reached</h3>"
        "Qanoon AI has answered too many questions today and reached its maximum server capacity. "
        "Please try again tomorrow when the quota resets!"
    )


@app.route('/')
def home(): return render_template('index.html')


@app.route('/consult', methods=['POST'])
def consult():
    data = request.json
    user_text = data.get('text', '')
    language_mode = data.get('lang', 'en') 
    
    print(f"🔍 Analyzing ({language_mode}): {user_text}")
    
    context = "No specific legal document found."
    
    # 💥 CRITICAL FIX: Load the brain ONLY when the user asks a question
    current_rag = get_rag()
    
    if current_rag:
        # OPTIMIZATION: Reduced k=8 to k=3 to prevent hitting Groq token limits
        docs = current_rag.search(user_text, k=3)
        if docs:
            context = ""
            for doc in docs:
                context += f"\n--- SOURCE: {doc['title']} ---\n{doc['text']}\n"

    # --- SYSTEM PROMPT (CONCISE GOVERNMENT ADVISOR) ---
    system_prompt = (
        "Role: You are Qanoon AI, an authoritative and professional legal advisor for Pakistani Law.\n"
        "Task: Provide a highly concise, government-style legal summary based STRICTLY on the provided text.\n\n"
        "CRITICAL RULES:\n"
        "1. MAX LENGTH: Keep the entire response under 4 sentences or 60 words to ensure rapid readability.\n"
        "2. TONE: Speak directly and officially. NEVER use phrases like 'According to the text' or 'The provided data says'.\n"
        "3. ACCURACY: Do not invent penalties. Use only what is provided.\n\n"
        "Format EXACTLY with these HTML tags (No Markdown):\n"
        "<h3>📜 Legal Overview</h3>\n"
        "[1 clear sentence summarizing the law.]\n"
        "<h3>⚖️ Penalties & Procedure</h3>\n"
        "[1-2 short bullet points using <ul><li> for specific punishments or fines.]\n"
        "<h3>📌 Official Reference</h3>\n"
        "<b>Source:</b> [Exact title/section from the text].\n\n"
        f"Language: {'Urdu' if language_mode == 'ur' else 'English'}."
    )
    
    full_prompt = f"{system_prompt}\nDATA:\n{context}\n\nQUERY: {user_text}"

    return Response(stream_with_context(generate_groq_response(full_prompt)), mimetype='text/plain')


# --- LAWYER DATABASE LOGIC ---
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
            print(f"⚠️ Warning: Database file not found at {LAWYERS_DB_PATH}")
            return jsonify([]) 
    except Exception as e:
        print(f"❌ Error reading lawyer DB: {e}")
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