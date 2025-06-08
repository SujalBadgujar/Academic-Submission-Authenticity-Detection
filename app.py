from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import zipfile
import shutil
from werkzeug.utils import secure_filename
import joblib
from utils.text_processor import TextProcessor
from utils.pdf_handler import PDFHandler
from utils.similarity import calculate_similarity_matrix
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'text_classifier.joblib')
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        flash('No files uploaded')
        return redirect(url_for('index'))
    
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        flash('No files selected')
        return redirect(url_for('index'))
    
    # Process uploaded files
    pdf_files = []
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], 'current_batch')
    
    # Clean up previous uploads
    if os.path.exists(upload_path):
        shutil.rmtree(upload_path)
    os.makedirs(upload_path)
    
    try:
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(upload_path, filename)
                file.save(filepath)
                
                # Handle ZIP files
                if filename.endswith('.zip'):
                    with zipfile.ZipFile(filepath, 'r') as zip_ref:
                        zip_ref.extractall(upload_path)
                    os.remove(filepath)
                    
                    # Find all PDFs in extracted content
                    for root, dirs, files in os.walk(upload_path):
                        for f in files:
                            if f.endswith('.pdf'):
                                pdf_files.append(os.path.join(root, f))
                else:
                    pdf_files.append(filepath)
        
        if not pdf_files:
            flash('No PDF files found')
            return redirect(url_for('index'))
        
        # Process PDFs
        results = process_pdfs(pdf_files)
        
        # Clean up uploads
        shutil.rmtree(upload_path)
        
        return render_template('results.html', results=results)
        
    except Exception as e:
        flash(f'Error processing files: {str(e)}')
        return redirect(url_for('index'))

def process_pdfs(pdf_files):
    pdf_handler = PDFHandler()
    text_processor = TextProcessor()
    
    results = {
        'files': [],
        'similarity_matrix': None,
        'average_ai_score': 0
    }
    
    texts = []
    
    for pdf_path in pdf_files:
        try:
            # Extract text from PDF
            text = pdf_handler.extract_text(pdf_path)
            texts.append(text)
            
            # Preprocess text for model
            processed_text = text_processor.preprocess_for_model(text)
            
            # Get AI detection score
            if model:
                ai_score = get_ai_score(processed_text)
            else:
                ai_score = 0.0
            
            results['files'].append({
                'name': os.path.basename(pdf_path),
                'ai_score': ai_score,
                'text_preview': text[:200] + '...' if len(text) > 200 else text
            })
            
        except Exception as e:
            results['files'].append({
                'name': os.path.basename(pdf_path),
                'ai_score': 0.0,
                'error': str(e)
            })
    
    # Calculate similarity matrix
    if len(texts) > 1:
        results['similarity_matrix'] = calculate_similarity_matrix(texts)
    
    # Calculate average AI score
    valid_scores = [f['ai_score'] for f in results['files'] if 'error' not in f]
    results['average_ai_score'] = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    
    return results

def get_ai_score(text):
    """Get AI-generated content probability from model"""
    try:
        # Get prediction probability
        proba = model.predict_proba([text])[0]
        # Return the probability of AI-generated text (assuming it's the positive class)
        return float(proba[1] * 100)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return 0.0

if __name__ == '__main__':
    app.run(debug=True)