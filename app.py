import os
import sys
import json
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from dotenv import load_dotenv
from google import genai  # THE NEW LIBRARY

# 1. SETUP PATHS & SECURITY
sys.path.append(os.getcwd()) 
load_dotenv()  # This loads the key from your .env file

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ ERROR: API Key not found! Make sure .env file exists.")

# 2. CONFIGURE THE NEW GEMINI CLIENT
# The new SDK uses a client instance rather than global configuration
client = genai.Client(api_key=api_key)

# 3. IMPORT RAG
rag = None
try:
    from backend.ai.rag_engine import rag
    print("✅ SUCCESS: AI Engine Loaded!")
except Exception as e:
    print(f"❌ ERROR: AI Engine Failed - {e}")

app = Flask(__name__)

# --- STREAMING GENERATOR (UPDATED FOR NEW SDK) ---
def generate_gemini_response(prompt):
    try:
        # The new method syntax is 'models.generate_content_stream'
        response = client.models.generate_content_stream(
            model='gemini-2.5-flash', # Using the latest Flash model
            contents=prompt
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"Error: {str(e)}"

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
        # Search more docs since Gemini handles large context easily
        docs = rag.search(user_text, k=5)
        if docs:
            context = ""
            for doc in docs:
                context += f"\n--- LEGAL REFERENCE ---\n{doc['text']}\n"

    # --- SYSTEM PROMPT ---
    system_prompt = (
        "You are Qanoon AI, a Pakistani Legal Assistant. "
        "Answer strictly based on the LEGAL REFERENCES provided below.\n"
        f"Output Language: {'URDU (Nastaliq)' if language_mode == 'ur' else 'ENGLISH'}.\n"
        "Format: Use bullet points and bold text for clarity.\n"
        "If the answer is not in the text, say you don't know."
    )

    full_prompt = f"{system_prompt}\nDATA:\n{context}\n\nQUERY: {user_text}"

    return Response(stream_with_context(generate_gemini_response(full_prompt)), mimetype='text/plain')

# --- LAWYER DATABASE LOGIC (UNCHANGED) ---
LAWYERS_DB_PATH = os.path.join("backend", "data", "raw", "lawyers_db.json")

def load_lawyers_db():
    try:
        if os.path.exists(LAWYERS_DB_PATH):
            with open(LAWYERS_DB_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ Error loading DB: {e}")
    return [] 

@app.route('/lawyers', methods=['GET'])
def get_lawyers():
    category = request.args.get('category', 'general').lower().strip()
    all_lawyers = load_lawyers_db()
    
    if not all_lawyers: return jsonify([])

    if category == 'general' or not category: 
        return jsonify(all_lawyers[:10])
    
    filtered = []
    for l in all_lawyers:
        tag_match = any(tag in category for tag in l.get('tags', []))
        spec_match = l.get('specialty', '').lower() in category or category in l.get('specialty', '').lower()
        if tag_match or spec_match:
            filtered.append(l)
    
    if not filtered:
        return jsonify(all_lawyers[:5])
        
    return jsonify(filtered)

if __name__ == '__main__':
    app.run(debug=True, port=5000)