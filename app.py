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

if not groq_api_key:
    print("❌ ERROR: GROQ_API_KEY not found in environment.")

# Initialize LLM with Groq
try:
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0.0,  
        api_key=groq_api_key,
        max_tokens=1024,
        max_retries=1 
    )
    print("⚡ SUCCESS: Groq AI Model Ready!")
except Exception as e:
    print(f"❌ ERROR: Groq Initialization Failed - {e}")

rag = None
try:
    from backend.ai.rag_engine import RAGEngine
    rag = RAGEngine()
    
    print("🔥 Forcing Local FAISS Brain to wake up...")
    is_awake = False
    while not is_awake:
        try:
            rag.embeddings.embed_query("wake up")
            is_awake = True
            print("✅ SUCCESS: Local FAISS Memory is fully awake!")
        except Exception:
            print("⏳ Model is still booting. Knocking again in 5 seconds...")
            time.sleep(5)
            
except Exception as e:
    print(f"❌ ERROR: Local AI Memory Failed - {e}")

def keep_brain_awake():
    while True:
        time.sleep(300) 
        if rag:
            try:
                rag.embeddings.embed_query("heartbeat ping")
                print("💓 [Heartbeat] Sent signal to keep Embedding Brain awake.")
            except Exception:
                pass 

threading.Thread(target=keep_brain_awake, daemon=True).start()

app = Flask(__name__)

def generate_groq_response(prompt):
    try:
        for chunk in llm.stream(prompt):
            if chunk.content:
                yield chunk.content
        return  

    except Exception as e:
        error_msg = str(e).lower()
        if '429' in error_msg or 'rate_limit' in error_msg:
            yield (
                "\n\n### ⏳ Whoa, Slow Down!\n"
                "**[Per-Minute Limit Reached]**\n"
                "I am currently analyzing a massive amount of legal documents for you! "
                "Please wait **60 seconds**, take a deep breath, and ask your question again. 🕰️"
            )
            return 
        elif '413' in error_msg or 'request too large' in error_msg:
            yield (
                "\n\n### ✂️ Query Too Complex\n"
                "Your question required reading too many laws at once! "
                "Please ask a shorter, more specific legal question. ⚖️"
            )
            return
        else:
            yield f"\n\n### ⚠️ System Interruption\nAn unexpected error occurred: {str(e)}"
            return

@app.route('/')
def home(): return render_template('index.html')

@app.route('/consult', methods=['POST'])
def consult():
    data = request.json
    user_text = data.get('text', '').strip()
    user_lang = data.get('lang', 'en') # Detect language

    context = ""
    if rag:
        try:
            docs = rag.search(user_text, k=5) 
            if docs:
                for doc in docs:
                    text_snippet = doc.get('text', '')[:600]
                    context += f"\nTEXT: {text_snippet}\n"
        except Exception as e:
             return Response(f"Memory Error: {str(e)}", mimetype='text/plain')

    # DYNAMIC LANGUAGE INSTRUCTION
    lang_instruction = ""
    if user_lang == 'ur':
        lang_instruction = (
            "CRITICAL: The user prefers URDU. You MUST respond in professional, accurate, and formal 'Adalti' Urdu. "
            "Keep legal Section numbers in English digits (e.g., Section 302) but translate the explanation perfectly. "
            "Ensure the Urdu is natural and authoritative."
        )
    else:
        lang_instruction = "The user prefers ENGLISH. Provide a professional legal response in English."

    system_prompt = (
        f"You are Qanoon AI, an elite Legal Consultant for Pakistani Law.\n{lang_instruction}\n\n"
        "### 🧠 1. INTENT EVALUATION:\n"
        "Evaluate the intent. If it is a greeting or general chat, respond warmly in the chosen language.\n"
        "If it is off-topic, respond with the [OFF-TOPIC] warning in the chosen language.\n\n"
        "### 🏛️ 2. VISUAL STYLE & CITATION RULES:\n"
        "- Base analysis STRICTLY on the DATA provided.\n"
        "- If DATA is missing, state '🛑 [DATA MISSING]' in the chosen language.\n"
        "- Structure with clear headers (###), bold text, and bullet points.\n"
        "- ONLY cite using Section/Article numbers from the text.\n"
        "- End with: `📜 Legal Authority: [Section]`"
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
        else: return jsonify([]) 
    except Exception: return jsonify([])
    if not all_lawyers: return jsonify([])
    if category == 'general' or not category: return jsonify(all_lawyers[:10])
    for lawyer in all_lawyers:
        lawyer_tags = [t.lower() for t in lawyer.get('tags', [])]
        lawyer_specialty = lawyer.get('specialty', '').lower()
        if category in lawyer_tags or category in lawyer_specialty:
            filtered_lawyers.append(lawyer)
    if not filtered_lawyers: return jsonify(all_lawyers[:5])
    return jsonify(filtered_lawyers)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))