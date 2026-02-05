# Qanoon AI - Pakistan's Legal Intelligence Assistant ⚖️

**Qanoon AI** is a cutting-edge, bilingual (English/Urdu) legal assistant designed to democratize access to legal information in Pakistan. Powered by **Retrieval-Augmented Generation (RAG)** and local Large Language Models (LLMs), it allows users to query Pakistani legal documents (such as the PPC and Constitution) and receive accurate, cited answers in real-time.

Additionally, the platform features a **Smart Lawyer Recommendation Engine** that connects users with top-tier verified legal professionals based on their specific case needs (e.g., Criminal, Family, Corporate).

---

## 🚀 Key Features

* **⚡ High-Speed AI**: Optimized to run using the **Gemini Flash-2.5 api key** instant responses.
* **🇵🇰 Bilingual Support**: Seamlessly toggles between English and Urdu (with native Nastaliq font support).
* **🧠 RAG Engine**: Context-aware answers derived strictly from uploaded PDF legal documents, ensuring accuracy over hallucination.
* **⚖️ Smart Lawyer Search**: Dynamically recommends verified lawyers from the *Chambers Asia-Pacific* directory based on the context of the user's query.
* **💬 Streaming Interface**: Real-time typewriter-style text streaming for a premium user experience.

---

## 🛠️ Tech Stack

* **Backend**: Python (Flask)
* **Frontend**: HTML5, CSS3 (Modern Dark Theme), JavaScript (ES6)
* **LLM Model**: `Gemini flash-2.5` (Optimized for speed and accuracy)
* **Data Processing**: `pdfplumber` (PDF Text Extraction), JSON
* **Search Logic**: Keyword matching & Tag-based filtering

---

## ⚙️ Installation & Setup Guide
Note: Download urdu-news-1M dataset from kaggle and save it in data/raw
### Prerequisites
* **Python 3.10+** installed.

2. Create Virtual Environment
```bash
python -m venv venv
```

# Windows:
```bash
venv\Scripts\activate
```

# Mac/Linux:
```bash
source venv/bin/activate
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```
▶️ Running the Application

1. **Process the data**
```bash
python backend/scraping/process_data.py
```

2. **Run the rag engine for AI memory**
```bash
python backend/ai/rag_engine.py
```

3. **Start the app**
```bash
python app.py
```

📸 Usage
Ask a Question: Type a legal scenario (e.g., "What is the punishment for theft?").

Toggle Language: Click the Urdu / English button to switch languages instantly.

Find a Lawyer: The sidebar will automatically update with lawyers relevant to your query (e.g., displaying Criminal defense lawyers if you ask about theft).

👤 Author
Muhammad Umair Ashraf Full Stack Developer & AI Engineer

⚠️ Disclaimer
Qanoon AI is an informational tool and does not constitute formal legal advice. Users should consult with a qualified legal professional for official counsel.