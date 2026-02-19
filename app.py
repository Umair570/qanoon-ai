import os
import sys
import json
import time
import threading
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

# --- 3. STRICT EAGER LOAD & WARMUP ---
print("🔌 Initializing Cloud Brain on Startup...")
rag = None
try:
    from backend.ai.rag_engine import RAGEngine
    rag = RAGEngine()
    
    # 🔥 STRICT BLOCKING WARMUP: Do not let the app start until HF is awake
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

# --- 4. THE HEARTBEAT THREAD (Prevents HF from ever sleeping) ---
def keep_brain_awake():
    """Silently pings the HF API every 5 minutes in the background."""
    while True:
        time.sleep(300) # Wait 5 minutes
        if rag:
            try:
                rag.embeddings.embed_query("heartbeat ping")
                print("💓 [Heartbeat] Sent signal to keep Hugging Face awake.")
            except Exception:
                pass # Ignore errors in the background

# Start the background heartbeat immediately
threading.Thread(target=keep_brain_awake, daemon=True).start()

app = Flask(__name__)

# --- STREAM GENERATOR WITH RATE LIMIT SAFETY ---
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
    language_mode = data.get('lang', 'en') 
    
    print(f"🔍 Analyzing ({language_mode}): {user_text}")
    
    context = "No specific legal document found."
    
    if rag:
        try:
            # We removed the KeyError catch because the heartbeat guarantees it is awake!
            docs = rag.search(user_text, k=3)
            if docs:
                context = ""
                for doc in docs:
                    context += f"\n--- SOURCE: {doc['title']} ---\n{doc['text']}\n"
        except Exception as e:
            def generic_error_message():
                yield f"<h3>⚠️ Memory Search Error</h3>An error occurred while searching the database: {str(e)}"
            return Response(stream_with_context(generic_error_message()), mimetype='text/plain')

    # --- SYSTEM PROMPT (MODERN, GEN-Z & SKIMMABLE) ---
    system_prompt = (
        "Role: You are Qanoon AI, a modern, highly engaging, and user-friendly legal advisor for Pakistani Law.\n"
        "Task: Answer the user's legal query based STRICTLY on the provided DATA, but make it incredibly easy to read for a modern/Gen-Z audience.\n\n"
        "🚨 CRITICAL GUARDRAILS: 🚨\n"
        "1. OFFENSIVE OR IRRELEVANT: If the query contains profanity, abuse, or is unrelated to Pakistani law, respond ONLY with: 'I am Qanoon AI, a professional legal assistant. I can only answer questions related to Pakistani law.'\n"
        "2. IDENTITY: If asked who you are or who created you, respond ONLY with: 'I am Qanoon AI, a legal advisor designed to assist with Pakistani legal queries.'\n"
        "3. NO GUESSING: If the DATA does not contain the answer, respond ONLY with: 'I am sorry, but I do not have specific information regarding this in my current legal records.'\n"
        "4. NO FILENAMES: NEVER output raw filenames like '.pdf'.\n\n"
        "💬 MODERN FORMATTING RULES: 💬\n"
        "- HOOK: Start with a conversational, friendly opening sentence (e.g., 'Here is the breakdown on...' or 'Here is what Pakistani law says about...').\n"
        "- SKIMMABLE BULLETS: Break down the rules or penalties using emojis and short bullet points. \n"
        "- BOLD THE IMPACT: Always use **bold text** for the most important parts, especially numbers, years in prison, or fine amounts.\n"
        "- THE BOTTOM LINE: Add a quick 1-sentence summary at the end wrapping up the vibe of the law.\n"
        "- TONE: Keep it punchy, relatable, and human. Drop the boring, robotic legal jargon where possible.\n"
        "- REFERENCE: End with a clean citation on a new line like this: '📖 *Reference: Section [Number]*'.\n\n"
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