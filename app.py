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
        temperature=0.0,  # 👈 THE FIX: 0.0 means ZERO creativity/hallucination. Just facts.
        model_name="llama-3.3-70b-versatile", # 👈 THE FIX: Double the TPM limit (12,000)
        api_key=groq_api_key,
        max_tokens=1024 
    )
    print("⚡ SUCCESS: Groq AI Model Ready!")
except Exception as e:
    print(f"❌ ERROR: Groq Initialization Failed - {e}")
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
                "Please wait **60 seconds**, take a deep breath, and ask your question again. 🕰️"
            )
            return  # 👈 CRITICAL: Stops the function from looping
            
        # Catch Token Overload (413)
        elif '413' in error_msg or 'request too large' in error_msg:
            yield (
                "\n\n### ✂️ Query Too Complex\n"
                "Your question required reading too many laws at once! "
                "Please ask a shorter, more specific legal question. ⚖️"
            )
            return
            
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

    context = ""
    if rag:
        try:
            # RAG still fetches data, but the LLM will decide whether to use it or ignore it.
            docs = rag.search(user_text, k=3) 
            if docs:
                for doc in docs:
                    # 'title' is passed for context, but the prompt forbids showing it to the user
                    context += f"\nACT: {doc.get('title')}\nTEXT: {doc.get('text')}\n"
        except Exception as e:
             return Response(f"Memory Error: {str(e)}", mimetype='text/plain')

    # THE DYNAMIC GATEKEEPER & STYLIST PROMPT
    system_prompt = (
        "You are Qanoon AI, an elite and prestigious Legal Consultant specializing in Pakistani Law.\n"
        "Your goal is to provide authoritative, visually structured, and fascinating legal guidance.\n\n"
        "### 🧠 1. INTENT EVALUATION (GATEKEEPER):\n"
        "Evaluate the USER QUERY before answering to determine their intent.\n"
        "- If it is a greeting or general conversation (e.g., hi, how are you, who are you), IGNORE the DATA and respond warmly: 'Greetings! I am Qanoon AI, a specialized legal assistant for Pakistani law. How can I assist you with your legal matters today?'\n"
        "- If the query is abusive, inappropriate, or completely unrelated to law, IGNORE the DATA and respond EXACTLY with: '🛑 **[OFF-TOPIC]** I am Qanoon AI, a professional legal assistant. I can only assist with matters related to Pakistani law.'\n"
        "- If it is a valid legal question, proceed to the rules below.\n\n"
        "### 🏛️ 2. VISUAL STYLE & CITATION RULES:\n"
        "- Base your legal analysis STRICTLY on the provided DATA.\n"
        "- If the DATA is irrelevant to the legal query, say: '🛑 **[DATA MISSING]** I don't have the specific legal sections in my database to answer this accurately.'\n"
        "- Structure your answer with clear, eye-catching headers (e.g., ### ⚖️ Legal Analysis) and bullet points.\n"
        "- Use **bold** text for important terms, penalties, and timeframes.\n"
        "- NEVER cite the 'Document Title' or PDF name.\n"
        "- ONLY cite using the specific Section or Article number found in the text (e.g., Section 302 of the PPC).\n"
        "- Always end your response with a clear citation line: `📜 Legal Authority: [Section Number/Name]`.\n"
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