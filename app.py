import google.generativeai as genai
import os
import uuid
import shutil
import hashlib
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from typing import List, Dict
from flask import Flask, render_template, request, session, abort
from werkzeug.utils import secure_filename
import mimetypes

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize the model
model = genai.GenerativeModel('gemini-2.0-flash')

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['VECTOR_FOLDER'] = 'vector_stores'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VECTOR_FOLDER'], exist_ok=True)

# Dictionary to track vector stores per session
vector_stores = {}

# Initialize embeddings model once for reuse
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Dictionary to store hashed documents to avoid re-embedding
document_hashes = {}

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def check_file_type(file_path):
    """Verify file is actually a PDF by checking MIME type"""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type != 'application/pdf':
        os.remove(file_path)  # Remove the suspicious file
        return False
    return True

def get_ai_response(prompt):
    response = model.generate_content(prompt)
    return response.text

def process_pdf(pdf_path: str) -> str:
    # Validate PDF before processing
    if not check_file_type(pdf_path):
        raise ValueError("Invalid PDF file")
        
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_text(text: str) -> List[str]:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_session_folder() -> str:
    """Get or create unique folder for this session"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    session_folder = os.path.join(app.config['VECTOR_FOLDER'], session['user_id'])
    os.makedirs(session_folder, exist_ok=True)
    return session_folder

def hash_document(text_chunk):
    """Create a hash of the text chunk for deduplication"""
    return hashlib.md5(text_chunk.encode()).hexdigest()

def create_vector_store(text_chunks: List[str], collection_name: str):
    """Create vector store in user-specific directory with hashing for efficiency"""
    session_folder = get_session_folder()
    
    # Generate a unique collection name for this session
    session_collection = f"{session['user_id']}_{collection_name}"
    
    # Check if this collection already exists for this session
    if session['user_id'] in vector_stores:
        # Return existing vector store
        return vector_stores[session['user_id']]
    else:
        # Hash the documents to avoid re-embedding duplicates
        unique_chunks = []
        for chunk in text_chunks:
            chunk_hash = hash_document(chunk)
            if chunk_hash not in document_hashes:
                unique_chunks.append(chunk)
                document_hashes[chunk_hash] = True
        
        # Create new vector store with the specified collection name
        vector_store = Chroma.from_texts(
            texts=unique_chunks if unique_chunks else text_chunks,
            embedding=embeddings,  # Reuse the global embeddings object
            persist_directory=session_folder,
            collection_name=session_collection
        )
        vector_store.persist()
        
        # Store in session-specific dictionary
        vector_stores[session['user_id']] = vector_store
        
        return vector_store

def get_relevant_chunks(query: str, vector_store) -> List[str]:

    results = vector_store.similarity_search(query, k=3)
    return [doc.page_content for doc in results]

def get_rag_response(query: str, context: List[str]) -> str:

    context_text = "\n".join(context)
    prompt = f"""You are a helpful assistant. Given the context below, answer the user's question in well-formatted HTML.

    Context:
    ---------------------
    {context_text}
    ---------------------

    Question: {query}

    Format your entire response using appropriate HTML tags for better readability:
    - Use <h3> for section headings
    - Use <p> for paragraphs
    - Use <ul> and <li> for lists
    - Use <code> for code snippets or technical terms
    - Use <strong> for emphasis
    - Use <br> for line breaks
    - Use <table>, <tr>, <th>, <td> for tabular data if needed

    IMPORTANT: Do NOT include any markdown code block markers (like ```html or ```) in your response. Return ONLY the raw HTML content.
    """
    
    response = get_ai_response(prompt)
    
    # Clean up any markdown code block markers if they still appear
    response = response.replace("```html", "").replace("```", "").strip()
    
    return response

def process_query(query: str, vector_store) -> str:

    relevant_chunks = get_relevant_chunks(query, vector_store)
    return get_rag_response(query, relevant_chunks)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Get session-specific vector store
    vector_store = vector_stores.get(session.get('user_id'))
    
    # Handle file uploads per user
    user_id = session.get('user_id', str(uuid.uuid4()))
    if 'user_id' not in session:
        session['user_id'] = user_id
    
    user_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
    os.makedirs(user_upload_folder, exist_ok=True)
    
    uploaded_filename = None
    
    # Check for user's uploaded files
    if os.path.exists(user_upload_folder):
        files = os.listdir(user_upload_folder)
        if files:
            # Get the most recently uploaded file
            uploaded_filename = files[-1]
    
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename:
                # Check if file type is allowed
                if not allowed_file(file.filename):
                    return render_template('index.html', 
                                         error="Only PDF files are allowed!")
                
                filename = secure_filename(file.filename)
                filepath = os.path.join(user_upload_folder, filename)
                file.save(filepath)
                
                try:
                    # Process PDF
                    text = process_pdf(filepath)
                    chunks = split_text(text)
                    vector_store = create_vector_store(chunks, "pdf_collection")
                    vector_stores[session['user_id']] = vector_store
                    
                    return render_template('index.html', 
                                          message="PDF processed successfully!",
                                          uploaded_filename=filename)
                except ValueError as e:
                    # Handle invalid PDF file
                    return render_template('index.html', 
                                          error=str(e))
                
        elif 'question' in request.form and vector_store:
            question = request.form['question']
            answer = process_query(question, vector_store)
            return render_template('index.html', 
                                  answer=answer,
                                  uploaded_filename=uploaded_filename)
            
    return render_template('index.html', uploaded_filename=uploaded_filename)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)
