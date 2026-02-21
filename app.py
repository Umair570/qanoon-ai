import os
import sys
import json
import time
import threading
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv

from langchain_groq import ChatGroq # 👈 Reverted to Groq

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
        model_name="llama-3.3-70b-versatile", # 👈 High accuracy 70B model
        temperature=0.0,  # 👈 0.0 means ZERO creativity/hallucination. Just facts.
        api_key=groq_api_key,
        max_tokens=1024,
        max_retries=1 # 👈 THE FIX: Stops silent sleep loops so it fails fast!
    )
    print("⚡ SUCCESS: Groq AI Model Ready!")
except Exception as e:
    print(f"❌ ERROR: Groq Initialization Failed - {e}")

rag = None
try:
    # This calls your RAGEngine class
    from backend.ai.rag_engine import RAGEngine
    rag = RAGEngine()
    
    # Render Memory-Safe Wakeup
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

# Keep-alive heartbeat (Critical for Hugging Face Inference API)
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
        # Stream the response directly to the user
        for chunk in llm.stream(prompt):
            if chunk.content:
                yield chunk.content
        return  # 👈 CRITICAL: Exits the generator successfully

    except Exception as e:
        error_msg = str(e).lower()
        
        # Catch Rate Limits (429) - Print ONCE and exit
        if '429' in error_msg or 'rate_limit' in error_msg:
            yield (
                "\n\n### ⏳ Whoa, Slow Down!\n"
                "**[Per-Minute Limit Reached]**\n"
                "I am currently analyzing a massive amount of legal documents for you! "
                "Please wait **60 seconds**, take a deep breath, and ask your question again. If still fails then daily limit reached. Try again tomorrow."
            )
            return  # 👈 CRITICAL: Stops the function from looping
            
        # Generic fallback
        else:
            yield f"\n\n### ⚠️ System Interruption\nAn unexpected error occurred: {str(e)}"
            return

@app.route('/')
def home(): return render_template('index.html')

@app.route('/consult', methods=['POST'])
def consult():
    data = request.json
    user_text = data.get('text', '').strip()
    user_lang = data.get('lang', 'en') # 👈 NEW: Detect language from frontend

    context = ""
    if rag:
        try:
            # 1. THE WIDE NET: Search k=5 to ensure critical laws are caught
            docs = rag.search(user_text, k=5) 
            if docs:
                for doc in docs:
                    # 2. THE SHORT TAIL: Aggressively chop text to only 600 characters. 
                    # 5 docs * 600 chars = 3,000 characters (Safely under Groq Token Limit)
                    text_snippet = doc.get('text', '')[:600]
                    context += f"\nTEXT: {text_snippet}\n"
        except Exception as e:
             return Response(f"Memory Error: {str(e)}", mimetype='text/plain')

    # THE BULLETPROOF DECISION TREE PROMPT
    if user_lang == 'ur':
        system_prompt = (
            "You are Qanoon AI, an elite Legal Consultant specializing in Pakistani Law.\n"
            "CRITICAL INSTRUCTION: The user prefers URDU. You MUST write your ENTIRE response in formal, professional 'Adalti' (Legal) Urdu.\n\n"
            "### 🧠 STEP 1: INTENT EVALUATION (DO NOT print this step)\n"
            "Analyze the user's query:\n"
            "1. If it is a greeting: Respond ONLY with 'السلام علیکم! میں قانون اے آئی ہوں، آپ کا قانونی معاون۔ میں آپ کی کیا مدد کر سکتا ہوں؟' and STOP.\n"
            "2. If the query is abusive, slang, or non-legal: Respond ONLY with '🛑 **[OFF-TOPIC]** میں صرف پاکستانی قانون سے متعلق سوالات کے جوابات دے سکتا ہوں۔' and STOP. Do NOT add any legal analysis or citations.\n"
            "3. If it is a valid legal question: Proceed to Step 2.\n\n"
            "### 🏛️ STEP 2: LEGAL FORMATTING (Only for valid legal questions)\n"
            "- Base your analysis STRICTLY on the provided DATA.\n"
            "- If DATA is missing/irrelevant, say: '🛑 **[DATA MISSING]** میرے پاس اس سوال کا جواب دینے کے لیے مخصوص قانونی حوالہ موجود نہیں ہے۔'\n"
            "- Use EXACTLY these two headers:\n"
            "### ⚖️ قانونی تجزیہ\n"
            "(Your detailed Urdu analysis here using bullet points. Keep Section numbers in English digits, e.g., Section 302)\n"
            "### 📜 قانونی حوالہ\n"
            "(List the specific Sections/Articles here, e.g., Section 380)\n"
            "- DO NOT add any extra citation lines at the very end. The 'قانونی حوالہ' section is your final conclusion."
        )
    else:
        system_prompt = (
            "You are Qanoon AI, an elite Legal Consultant specializing in Pakistani Law.\n"
            "CRITICAL INSTRUCTION: The user prefers ENGLISH. You must write your entire response in professional English.\n\n"
            "### 🧠 STEP 1: INTENT EVALUATION (DO NOT print this step)\n"
            "Analyze the user's query:\n"
            "1. If it is a greeting: Respond ONLY with 'Greetings! I am Qanoon AI, a specialized legal assistant for Pakistani law. How can I assist you today?' and STOP.\n"
            "2. If the query is abusive, slang, or non-legal: Respond ONLY with '🛑 **[OFF-TOPIC]** I am Qanoon AI, a professional legal assistant. I can only assist with matters related to Pakistani law.' and STOP. Do NOT add any legal analysis or citations.\n"
            "3. If it is a valid legal question: Proceed to Step 2.\n\n"
            "### 🏛️ STEP 2: LEGAL FORMATTING (Only for valid legal questions)\n"
            "- Base your analysis STRICTLY on the provided DATA.\n"
            "- If DATA is missing/irrelevant, say: '🛑 **[DATA MISSING]** I don't have the specific legal sections in my database to answer this accurately.'\n"
            "- Use EXACTLY these two headers:\n"
            "### ⚖️ Legal Analysis\n"
            "(Your detailed analysis here using bullet points)\n"
            "### 📜 Legal Authority\n"
            "(List the specific Sections/Articles here, e.g., Section 302 of the PPC)\n"
            "- DO NOT add any extra citation lines at the very end. The 'Legal Authority' section is your final conclusion."
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