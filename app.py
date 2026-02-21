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
        model_name="llama3-70b-8192", 
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
            yield "\n\n### ⏳ Limit Reached\nPlease wait 60 seconds, take a deep breath, and ask again. If still fails then daily limit is reached. Try again tomorrow."
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
    user_lang = data.get('lang', 'en') 

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

    # 🛡️ THE FIXED PROMPT GATEKEEPER (With "No Fourth Wall" Rule)
    if user_lang == 'ur':
        lang_instruction = (
            "CRITICAL INSTRUCTION: User prefers URDU. Write ENTIRE response in formal 'Adalti' (Legal) Urdu.\n\n"
            "### 🧠 STEP 1: INTENT EVALUATION (DO NOT print this step)\n"
            "- If query is a greeting: Respond ONLY with 'السلام علیکم! میں قانون اے آئی ہوں، آپ کا قانونی معاون۔ میں آپ کی کیا مدد کر سکتا ہوں؟' and STOP.\n"
            "- If query is abusive/off-topic: Respond ONLY with '🛑 **[OFF-TOPIC]** میں صرف پاکستانی قانون سے متعلق سوالات کے جوابات دے سکتا ہوں۔' and STOP.\n"
            "- If valid legal question: Proceed to Step 2.\n\n"
            "### 🏛️ STEP 2: LEGAL FORMATTING (For valid questions ONLY)\n"
            "- RULE 1: DO NOT include greetings here. Start directly with the analysis.\n"
            "- RULE 2: NEVER break character. NEVER use phrases like 'according to the provided text', 'in the data', or 'مہیا کیے گئے متن کے مطابق'. State the law authoritatively as an expert.\n"
            "- RULE 3: Use EXACTLY these headers:\n"
            "### ⚖️ قانونی تجزیہ\n"
            "(Your Urdu analysis here using bullet points. Keep Section numbers in English digits, e.g., Section 302)\n"
            "### 📜 قانونی حوالہ\n"
            "(List specific Sections here)\n"
            "- RULE 4: DO NOT add any extra text or citation lines at the very end. The 'قانونی حوالہ' section is your conclusion."
        )
    else:
        lang_instruction = (
            "CRITICAL INSTRUCTION: User prefers ENGLISH. Write ENTIRE response in professional English.\n\n"
            "### 🧠 STEP 1: INTENT EVALUATION (DO NOT print this step)\n"
            "- If query is a greeting: Respond ONLY with 'Greetings! I am Qanoon AI, a specialized legal assistant for Pakistani law. How can I assist you today?' and STOP.\n"
            "- If query is abusive/off-topic: Respond ONLY with '🛑 **[OFF-TOPIC]** I can only assist with matters related to Pakistani law.' and STOP.\n"
            "- If valid legal question: Proceed to Step 2.\n\n"
            "### 🏛️ STEP 2: LEGAL FORMATTING (For valid questions ONLY)\n"
            "- RULE 1: DO NOT include greetings here. Start directly with the analysis.\n"
            "- RULE 2: NEVER break character. NEVER use phrases like 'according to the provided text', 'in the data', or 'the documents state'. State the law authoritatively as an expert.\n"
            "- RULE 3: Use EXACTLY these headers:\n"
            "### ⚖️ Legal Analysis\n"
            "(Your English analysis here using bullet points)\n"
            "### 📜 Legal Authority\n"
            "(List specific Sections here)\n"
            "- RULE 4: DO NOT add any extra text or citation lines at the very end. The 'Legal Authority' section is your conclusion."
        )

    system_prompt = (
        f"You are Qanoon AI, an elite Legal Consultant for Pakistani Law.\n{lang_instruction}\n\n"
        "### DATA RULES:\n"
        "- Base analysis STRICTLY on the DATA provided, but do not mention the data itself.\n"
        "- If the answer cannot be found in the DATA, state '🛑 [DATA MISSING]' in the chosen language."
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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))