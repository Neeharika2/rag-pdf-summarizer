# RAG Application

A Retrieval-Augmented Generation (RAG) application that allows users to upload documents and ask questions about their content.

## Features

- Document upload and management
- Natural language question answering based on uploaded documents
- Clean and responsive user interface
- Real-time answers

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

- `app.py`: Main application file
- `static/`: Static files (CSS, JavaScript)
- `templates/`: HTML templates
- `models/`: Contains ML models for RAG functionality
- `utils/`: Utility functions

## Requirements

- Python 3.8+
- Flask
- Vector database (e.g., FAISS, Pinecone)
- LLM integration (e.g., OpenAI API)

## License

[Add your license here]
