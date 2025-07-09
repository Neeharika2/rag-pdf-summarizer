# 🤖 RAG-Powered Document Question Answering App

An AI-powered Retrieval-Augmented Generation (RAG) application that allows users to **upload custom documents (PDFs)** and interactively **ask questions** about their content using **LLMs**, **vector embeddings**, and **natural language processing**.

---

## 🚀 Features

- 📄 Upload and manage PDF documents
- 🧠 Ask natural language questions based on document content
- 🔎 Embedding-based document chunking and semantic retrieval using **FAISS**
- 💬 Context-aware LLM responses via **OpenAI / HuggingFace Transformers**
- ⚙️ Powered by **LangChain** for modular RAG pipeline
- 📱 Clean, responsive web UI built with Flask & HTML templates

---

## 🧰 Tech Stack

- **Python 3.8+**
- **Flask** (for web server and routing)
- **LangChain** (for chaining document retriever + LLM)
- **Vector DB:** FAISS (can be extended to Pinecone, ChromaDB, etc.)
- **LLM Providers:** OpenAI API / HuggingFace Transformers
- **PDF Parsing:** PyMuPDF / pdfminer
- **Env Handling:** `dotenv`

---

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Neeharika2/rag-pdf-summarizer
cd rag_application
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (if needed):
```bash
# Create .env file with required variables
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Upload documents using the interface on the left panel

4. Ask questions about your documents in the question box at the bottom of the right panel

## Project Structure

    rag_application/
    │
    ├── app.py                # Flask application entry point
    ├── rag_utils.py          # LangChain + RAG setup (retriever, prompt chain)
    ├── templates/
    │   └── index.html        # UI template
    ├── static/
    │   └── style.css         # Custom CSS
    ├── uploads/              # Uploaded PDF storage
    ├── requirements.txt
    ├── .env

## How It Works (Behind the Scenes)
- PDF Upload: User uploads a document
- Chunking & Embedding: PDF is split into chunks & embedded using OpenAI/HuggingFace embeddings  
- Vector Storage: Embeddings are stored/retrieved using FAISS 
- Query Handling: User question is embedded and top relevant chunks are retrieved  
- LLM Generation: LLM generates a contextual, semantic answer

## Requirements

- Python 3.8+
- Flask
- Vector database (e.g., FAISS, Pinecone)
- LLM integration (e.g., OpenAI API)

