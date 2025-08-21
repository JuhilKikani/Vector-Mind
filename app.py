import os
import io
import PyPDF2
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Directory to temporarily store uploaded PDFs
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Google Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.error("GOOGLE_API_KEY not found in environment variables. Please set it.")
    exit("GOOGLE_API_KEY not set. Please refer to the documentation to set it up.")

genai.configure(api_key=GOOGLE_API_KEY)

faiss_index = None
document_chunks = []
embedding_model = None 

def initialize_embedding_model():
    """Initializes the Sentence Transformer model if not already initialized."""
    global embedding_model
    if embedding_model is None:
        try:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("SentenceTransformer model 'all-MiniLM-L6-v2' loaded.")
        except Exception as e:
            logging.error(f"Error loading SentenceTransformer model: {e}")
            raise

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or ""
        logging.info(f"Text extracted from {pdf_path}")
    except Exception as e:
        logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""
    return text

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    """
    Splits text into smaller, overlapping chunks.

    Args:
        text (str): The input text.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        list: A list of text chunks.
    """
    chunks = []
    if not text:
        return chunks

    words = text.split()
    current_chunk = []
    current_length = 0

    for word in words: 
        if current_length + len(word) + 1 > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Start new chunk with overlap
            overlap_words = " ".join(current_chunk[-int(chunk_overlap / 5):]) 
            current_chunk = overlap_words.split() if overlap_words else []
            current_length = sum(len(w) for w in current_chunk) + len(current_chunk) -1 if current_chunk else 0

        current_chunk.append(word)
        current_length += len(word) + 1 

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logging.info(f"Text chunked into {len(chunks)} chunks.")
    return chunks

def get_embeddings(texts, task_type):
    """
    Generates embeddings for a list of texts using Google Gemini embedding model
    or a local SentenceTransformer model.

    Args:
        texts (list[str]): A list of texts to embed.
        task_type (str): The task type for the embedding model (e.g., "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY").

    Returns:
        np.array: A NumPy array of embeddings.
    """
    if not texts:
        return np.array([])

    embeddings = []
    try:
        initialize_embedding_model()
        embeddings = embedding_model.encode(texts)
        logging.info(f"Generated {len(embeddings)} embeddings using SentenceTransformer.")

    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        if embedding_model is None: 
            initialize_embedding_model()
        embeddings = embedding_model.encode(texts) 
        logging.info(f"Generated {len(embeddings)} embeddings using SentenceTransformer after error.")
    
    return np.array(embeddings).astype('float32') 

def build_faiss_index(embeddings):
    """
    Builds a FAISS index from a NumPy array of embeddings.

    Args:
        embeddings (np.array): A NumPy array of embeddings.

    Returns:
        faiss.Index: A FAISS index.
    """
    if embeddings.shape[0] == 0:
        logging.warning("No embeddings provided to build FAISS index.")
        return None
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    logging.info(f"FAISS index built with {index.ntotal} vectors of dimension {dimension}.")
    return index

def retrieve_relevant_chunks(query_embedding, k=5):
    """
    Retrieves the most relevant document chunks based on a query embedding.

    Args:
        query_embedding (np.array): The embedding of the query.
        k (int): The number of top relevant chunks to retrieve.

    Returns:
        list: A list of relevant text chunks.
    """
    global faiss_index, document_chunks
    if faiss_index is None or faiss_index.ntotal == 0:
        logging.warning("FAISS index is not built or empty.")
        return []

    
    query_embedding = query_embedding.reshape(1, -1)
    
    # Perform similarity search
    D, I = faiss_index.search(query_embedding, k) 
    
    relevant_chunks = [document_chunks[i] for i in I[0] if i < len(document_chunks)]
    logging.info(f"Retrieved {len(relevant_chunks)} relevant chunks.")
    return relevant_chunks

def generate_answer(question, relevant_chunks):
    """
    Generates an answer to a question using the Google Gemini Flash model,
    augmented with relevant document chunks.

    Args:
        question (str): The user's question.
        relevant_chunks (list[str]): A list of text chunks retrieved from the document.

    Returns:
        str: The generated answer.
    """
    if not relevant_chunks:
        logging.warning("No relevant chunks provided for answer generation.")
        return "I couldn't find enough information in the document to answer that question. Please try uploading a different document or rephrasing your question."

    context = "\n".join(relevant_chunks)
    prompt = f"""
    You are an AI assistant tasked with answering questions based on the provided document excerpts.
    If the question cannot be answered from the document, state that you don't have enough information.

    Document excerpts:
    \"\"\"
    {context}
    \"\"\"

    Question: {question}

    Answer:
    """
    logging.info("Sending prompt to Gemini API...")
    try:
        
        model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20') 
        response = model.generate_content(prompt)
        answer = response.text
        logging.info("Answer generated successfully.")
        return answer
    except Exception as e:
        logging.error(f"Error calling Gemini API for answer generation: {e}")
        return "An error occurred while generating the answer. Please try again."

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """
    Handles PDF file uploads, extracts text, chunks it, and builds a FAISS index.
    """
    global faiss_index, document_chunks

    if 'pdfFile' not in request.files:
        logging.warning("No pdfFile part in the request.")
        return jsonify({"success": False, "message": "No file part"}), 400

    file = request.files['pdfFile']
    if file.filename == '':
        logging.warning("No selected file.")
        return jsonify({"success": False, "message": "No selected file"}), 400

    if file and file.filename.endswith('.pdf'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        try:
            file.save(file_path)
            logging.info(f"File saved to {file_path}")

            text = extract_text_from_pdf(file_path)
            if not text:
                os.remove(file_path) 
                return jsonify({"success": False, "message": "Failed to extract text from PDF."}), 500

            chunks = chunk_text(text)
            if not chunks:
                os.remove(file_path) 
                return jsonify({"success": False, "message": "No content found in PDF to process."}), 500

            document_chunks = chunks
            embeddings = get_embeddings(document_chunks, "RETRIEVAL_DOCUMENT")
            
            if embeddings.shape[0] == 0:
                os.remove(file_path) 
                return jsonify({"success": False, "message": "Failed to generate embeddings for the document."}), 500

            faiss_index = build_faiss_index(embeddings)
            
            os.remove(file_path) 

            return jsonify({"success": True, "message": f"PDF processed. {len(document_chunks)} chunks indexed."})
        except Exception as e:
            logging.error(f"Error during PDF upload and processing: {e}")
            
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"success": False, "message": f"Error processing PDF: {e}"}), 500
    else:
        logging.warning("Invalid file type uploaded.")
        return jsonify({"success": False, "message": "Invalid file type. Please upload a PDF."}), 400

@app.route('/ask_question', methods=['POST'])
def ask_question():
    """
    Handles user questions, retrieves relevant chunks, and generates an answer.
    """
    if faiss_index is None or faiss_index.ntotal == 0:
        return jsonify({"success": False, "message": "No document indexed yet. Please upload a PDF first."}), 400

    question = request.json.get('question')
    if not question:
        return jsonify({"success": False, "message": "No question provided."}), 400

    try:
        query_embedding = get_embeddings([question], "RETRIEVAL_QUERY")[0] # Get embedding for the question
        relevant_chunks = retrieve_relevant_chunks(query_embedding)
        answer = generate_answer(question, relevant_chunks)
        return jsonify({"success": True, "answer": answer})
    except Exception as e:
        logging.error(f"Error during question answering: {e}")
        return jsonify({"success": False, "message": f"Error answering question: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

