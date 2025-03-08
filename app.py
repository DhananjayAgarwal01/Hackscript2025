from flask import Flask, render_template, request, flash, redirect, jsonify, url_for
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from joblib import load
import warnings
from werkzeug.utils import secure_filename
import os
import cv2
import pytesseract
from PIL import Image
import PyPDF2  # Replace fitz with PyPDF2
import hashlib
from pathlib import Path
import re
import json
import sys
import platform

warnings.filterwarnings('ignore')

def initialize_tesseract():
    """Initialize Tesseract with proper path detection"""
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\dhana\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
        # Add your actual installation path here if different
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"Found Tesseract at: {path}")
            return True
    
    raise RuntimeError(
        "Tesseract not found. Please install it from: "
        "https://github.com/UB-Mannheim/tesseract/wiki"
    )

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
PAN_PATTERN = re.compile(r'^[A-Z]{5}[0-9]{4}[A-Z]$')
PAN_ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'pdf'}

PAN_ZONES = {
    'name': (50, 100, 400, 150),  # Example coordinates (x1, y1, x2, y2)
    'pan_number': (50, 200, 400, 250),
    'dob': (50, 150, 400, 200),
    'signature': (50, 300, 400, 350)
}

app = Flask(__name__, template_folder='templates')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///documents.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
db = SQLAlchemy(app)

# Add after app initialization and before database models
@app.template_filter('from_json')
def from_json(value):
    try:
        if isinstance(value, str):
            return json.loads(value)
        return value
    except (json.JSONDecodeError, TypeError):
        return {}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    initialize_tesseract()
except RuntimeError as e:
    print(f"Warning: {str(e)}")

# ********************************** Database Models **********************************
class ContactMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(10), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    document_type = db.Column(db.String(50), nullable=False)
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    hash_value = db.Column(db.String(64), nullable=False)
    forgery_score = db.Column(db.Float)
    analysis_result = db.Column(db.Text)
    document_metadata = db.Column(db.String(200))

class AnalysisResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    analysis_type = db.Column(db.String(50), nullable=False)
    result = db.Column(db.Text, nullable=False)
    confidence_score = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class PanVerification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pan_number = db.Column(db.String(10), nullable=False)
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    verification_status = db.Column(db.String(20), nullable=False)
    confidence_score = db.Column(db.Float)
    extracted_text = db.Column(db.Text)
    filename = db.Column(db.String(255))
    extracted_name = db.Column(db.String(100))
    extracted_dob = db.Column(db.String(20))
    face_detected = db.Column(db.Boolean, default=False)
    signature_detected = db.Column(db.Boolean, default=False)
    manipulated_score = db.Column(db.Float, default=0.0)

# ********************************** Main Routes **********************************
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/contact', methods=['POST'])
def contact():
    try:
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        description = request.form['description']

        new_message = ContactMessage(
            name=name,
            email=email,
            phone=phone,
            description=description
        )

        db.session.add(new_message)
        db.session.commit()

        flash('Thank you for your message! We will get back to you soon.', 'success')
        return redirect(url_for('home'))
    except Exception as e:
        flash('An error occurred while sending your message. Please try again.', 'error')
        return redirect(url_for('home'))

# ********************************** Prediction Functions **********************************
def load_data():
    try:
        df = pd.read_csv('static/models/ideathon_dataset.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def load_models():
    models = {}
    try:
        for commodity in ['onion', 'potato', 'tomato']:
            model_path = f'static/models/{commodity}_model.joblib'
            models[commodity] = load(model_path)
    except Exception as e:
        print(f"Error loading models: {e}")
    return models

# Initialize models
commodity_models = load_models()

# Base prices for wholesale
pulse_base_prices = {
    'gram': 60,
    'tur': 85,
    'urad': 90,
    'moong': 95,
    'masur': 80
}

# ********************************** Prediction Routes **********************************
@app.route('/predict_vegetables')
def predict_vegetables_form():
    return render_template('predict_vegetables.html')

@app.route('/predict_wholesale')
def predict_wholesale_form():
    return render_template('predict_wholesale.html')

@app.route('/predict_vegetables', methods=['POST'])
def predict_vegetables():
    try:
        commodity = request.form['commodity']
        date_str = request.form['date']
        market = request.form['market']
        quantity = float(request.form['quantity'])
        
        date = datetime.strptime(date_str, '%Y-%m-%d')
        
        df = load_data()
        if df is None:
            return jsonify({'error': 'Unable to load historical data'}), 500
            
        features = pd.DataFrame({
    'year': [date.year],
    'month': [date.month],
    'week': [date.isocalendar()[1]],
    f'lag_1_{commodity}': [df[commodity].iloc[-1]],
    f'rolling_mean_7_{commodity}': [df[commodity].rolling(window=7).mean().iloc[-1]],
    # Add missing features here (check model.feature_names_in_)
})

        
        try:
            model = load('static/models/final_retailprice_prediction_model.joblib')
            predicted_price = model.predict(features)[0]
            confidence = min(95, int(model.feature_importances_.mean() * 100))
            
            historical_dates = df.tail(30)['date'].dt.strftime('%Y-%m-%d').tolist()
            historical_prices = df[commodity].tail(30).tolist()
            
            future_dates = [(date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
            future_features = pd.DataFrame({
                'year': [(date + timedelta(days=i)).year for i in range(7)],
                'month': [(date + timedelta(days=i)).month for i in range(7)],
                'week': [(date + timedelta(days=i)).isocalendar()[1] for i in range(7)],
                f'lag_1_{commodity}': [df[commodity].iloc[-1]] * 7,
                f'rolling_mean_7_{commodity}': [df[commodity].rolling(window=7).mean().iloc[-1]] * 7
            })
            future_prices = model.predict(future_features).tolist()
            
            avg_price = df[commodity].mean()
            price_trend = 'Increasing' if predicted_price > avg_price else 'Decreasing'
            volatility = 'High' if abs(predicted_price - avg_price) > avg_price * 0.2 else 'Moderate'
            
            market_insights = f"The {commodity} prices are showing a {price_trend.lower()} trend. "
            market_insights += f"Price volatility is {volatility.lower()}. "
            
            total_price = predicted_price * quantity
            
            additional_insights = [
                f"Historical average price: ₹{avg_price:.2f}/kg",
                f"Current market trend: {price_trend}",
                f"Price volatility: {volatility}",
                f"Total price for {quantity}kg: ₹{total_price:.2f}",
                f"Recommended action: {'Stock up' if price_trend == 'Increasing' else 'Regular purchase'}"
            ]
            
            return render_template('prediction_result.html',
                                commodity=commodity,
                                date=date_str,
                                market=market,
                                quantity=quantity,
                                unit='kg',
                                predicted_price=predicted_price,
                                total_price=total_price,
                                confidence=confidence,
                                dates=historical_dates + future_dates,
                                historical_prices=historical_prices + future_prices,
                                predicted_prices=[None] * len(historical_dates) + future_prices,
                                market_insights=market_insights,
                                additional_insights=additional_insights)
                                
        except Exception as e:
            return jsonify({'error': f'Error loading model: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_wholesale', methods=['POST'])
def predict_wholesale():
    try:
        commodity = request.form['commodity']
        date_str = request.form['date']
        market = request.form['market']
        quantity = float(request.form['quantity'])
        
        date = datetime.strptime(date_str, '%Y-%m-%d')
        
        df = pd.read_csv('static/models/ideathon_wholesale_dataset.csv')  # Load wholesale dataset
        df['date'] = pd.to_datetime(df['date'])
        
        if df is None:
            return jsonify({'error': 'Unable to load historical data'}), 500
        
        features = pd.DataFrame({
            'year': [date.year],
            'month': [date.month],
            'week': [date.isocalendar()[1]],
            f'lag_1_{commodity}': [df[commodity].iloc[-1]],
            f'rolling_mean_7_{commodity}': [df[commodity].rolling(window=7).mean().iloc[-1]]
        })
        
        try:
            model = load(f'static/models/final_wholesaleprice_prediction_model.joblib')
            predicted_price = model.predict(features)[0]
            confidence = min(95, int(model.feature_importances_.mean() * 100))
            
            historical_dates = df.tail(30)['date'].dt.strftime('%Y-%m-%d').tolist()
            historical_prices = df[commodity].tail(30).tolist()
            
            future_dates = [(date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
            future_features = pd.DataFrame({
                'year': [(date + timedelta(days=i)).year for i in range(7)],
                'month': [(date + timedelta(days=i)).month for i in range(7)],
                'week': [(date + timedelta(days=i)).isocalendar()[1] for i in range(7)],
                f'lag_1_{commodity}': [df[commodity].iloc[-1]] * 7,
                f'rolling_mean_7_{commodity}': [df[commodity].rolling(window=7).mean().iloc[-1]] * 7
            })
            future_prices = model.predict(future_features).tolist()
            
            avg_price = df[commodity].mean()
            price_trend = 'Increasing' if predicted_price > avg_price else 'Decreasing'
            volatility = 'High' if abs(predicted_price - avg_price) > avg_price * 0.2 else 'Moderate'
            
            market_insights = f"The {commodity} prices are showing a {price_trend.lower()} trend. "
            market_insights += f"Price volatility is {volatility.lower()}. "
            
            total_price = predicted_price * quantity
            
            additional_insights = [
                f"Historical average price: ₹{avg_price:.2f}/kg",
                f"Current market trend: {price_trend}",
                f"Price volatility: {volatility}",
                f"Total price for {quantity}kg: ₹{total_price:.2f}",
                f"Recommended action: {'Stock up' if price_trend == 'Increasing' else 'Regular purchase'}"
            ]
            
            return render_template('prediction_result.html',
                                commodity=commodity,
                                date=date_str,
                                market=market,
                                quantity=quantity,
                                unit='kg',
                                predicted_price=predicted_price,
                                total_price=total_price,
                                confidence=confidence,
                                dates=historical_dates + future_dates,
                                historical_prices=historical_prices + future_prices,
                                predicted_prices=[None] * len(historical_dates) + future_prices,
                                market_insights=market_insights,
                                additional_insights=additional_insights)
                                
        except Exception as e:
            return jsonify({'error': f'Error loading model: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ********************************** Routes **********************************
@app.route('/upload', methods=['GET', 'POST'])
def upload_document():
    if request.method == 'POST':
        if 'document' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['document']
        document_type = request.form.get('document_type')
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Calculate file hash
            file_hash = calculate_file_hash(file_path)
            
            # Check if document with same hash exists
            existing_doc = Document.query.filter_by(hash_value=file_hash).first()
            if existing_doc:
                flash('This document has already been analyzed', 'warning')
                return redirect(url_for('view_result', document_id=existing_doc.id))
            
            # Analyze document
            analysis_results = analyze_document(file_path, document_type)
            
            # Save to database
            new_document = Document(
                filename=filename,
                document_type=document_type,
                hash_value=file_hash,
                forgery_score=analysis_results['forgery_probability'],
                analysis_result=json.dumps(analysis_results),  # Convert to JSON string
                metadata=json.dumps(analysis_results.get('metadata_analysis', {}))
            )
            db.session.add(new_document)
            db.session.commit()
            
            # Save detailed analysis results
            for analysis_type, result in analysis_results.items():
                if isinstance(result, dict):
                    analysis_entry = AnalysisResult(
                        document_id=new_document.id,
                        analysis_type=analysis_type,
                        result=json.dumps(result),  # Convert to JSON string
                        confidence_score=0.9  # Default confidence score
                    )
                    db.session.add(analysis_entry)
            
            db.session.commit()
            
            return redirect(url_for('view_result', document_id=new_document.id))
        
        flash('Invalid file type', 'error')
        return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/result/<int:document_id>')
def view_result(document_id):
    document = Document.query.get_or_404(document_id)
    analysis_results = AnalysisResult.query.filter_by(document_id=document_id).all()
    
    return render_template('result.html', 
                         document=document,
                         analysis_results=analysis_results)

@app.route('/history')
def view_history():
    documents = Document.query.order_by(Document.upload_date.desc()).all()
    return render_template('history.html', documents=documents)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/verify-pan', methods=['GET'])
def verify_pan_form():
    return render_template('verify_pan.html')

@app.route('/verify-pan', methods=['POST'])
def verify_pan():
    try:
        if 'pan_file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)

        file = request.files['pan_file']
        pan_number = request.form.get('pan_number', '').strip().upper()

        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if not is_valid_pan(pan_number):
            flash('Invalid PAN number format', 'error')
            return redirect(request.url)

        if file and file.filename.lower().endswith(tuple(PAN_ALLOWED_EXTENSIONS)):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Extract all details
            result = extract_pan_details(file_path)
            
            # Verify PAN number
            verification_status = 'VERIFIED' if result.get('pan_number') == pan_number else 'FAILED'
            
            # Advanced verification checks
            if verification_status == 'VERIFIED':
                if result.get('manipulated_score', 0) > 0.7:
                    verification_status = 'SUSPICIOUS'
                elif not result.get('face_detected'):
                    verification_status = 'INCOMPLETE'

            # Save verification result
            verification = PanVerification(
                pan_number=pan_number,
                verification_status=verification_status,
                confidence_score=result.get('confidence', 0.0),
                extracted_text=result.get('full_text', ''),
                filename=filename,
                extracted_name=result.get('extracted_name', ''),
                extracted_dob=result.get('extracted_dob', ''),
                face_detected=result.get('face_detected', False),
                signature_detected=result.get('extracted_signature', False) != '',
                manipulated_score=result.get('manipulated_score', 0.0)
            )
            db.session.add(verification)
            db.session.commit()

            # Clean up
            os.remove(file_path)

            return render_template('pan_result.html',
                                verification=verification,
                                result=result)
        
        flash('Invalid file type', 'error')
        return redirect(request.url)

    except Exception as e:
        flash(f'Error during verification: {str(e)}', 'error')
        return redirect(request.url)

@app.route('/pan-history')
def pan_verification_history():
    verifications = PanVerification.query.order_by(PanVerification.upload_date.desc()).all()
    return render_template('pan_history.html', verifications=verifications)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_file_hash(file_path):
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def analyze_document(file_path, document_type):
    """Analyze document for potential forgery"""
    results = {
        'content_analysis': {},
        'structural_analysis': {},
        'metadata_analysis': {},
        'forgery_probability': 0.0
    }
    
    # Extract text using OCR for images or PDF text extraction
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(file_path)
        text = pytesseract.image_to_string(img)
    else:  # PDF
        try:
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                
                # Get PDF metadata
                metadata = pdf_reader.metadata
                if metadata:
                    results['metadata_analysis'] = {
                        'author': metadata.get('/Author', ''),
                        'creator': metadata.get('/Creator', ''),
                        'producer': metadata.get('/Producer', ''),
                        'creation_date': metadata.get('/CreationDate', ''),
                        'modification_date': metadata.get('/ModDate', '')
                    }
        except Exception as e:
            print(f"PDF processing error: {str(e)}")
            text = ""
    
    # Basic content analysis
    results['content_analysis']['text_length'] = len(text)
    results['content_analysis']['has_suspicious_patterns'] = False
    
    # Structural analysis for images
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(file_path)
        # Error Level Analysis
        ela_score = perform_ela(file_path)
        results['structural_analysis']['ela_score'] = ela_score
    
    # Calculate overall forgery probability
    forgery_indicators = []
    if results['content_analysis'].get('has_suspicious_patterns', False):
        forgery_indicators.append(0.3)
    if results.get('structural_analysis', {}).get('ela_score', 0) > 50:
        forgery_indicators.append(0.4)
    
    results['forgery_probability'] = sum(forgery_indicators) if forgery_indicators else 0.1
    
    return results

def perform_ela(image_path):
    """Perform Error Level Analysis on image"""
    try:
        QUALITY = 90
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_ela.jpg")
        
        # Read original image
        original = Image.open(image_path)
        # Save with specific quality
        original.save(temp_path, 'JPEG', quality=QUALITY)
        
        # Read both images as numpy arrays
        original_array = np.array(Image.open(image_path).convert('L'))
        resaved_array = np.array(Image.open(temp_path).convert('L'))
        
        # Calculate difference and amplify
        diff = np.absolute(original_array - resaved_array)
        diff = diff * 255.0 / diff.max()  # Normalize to 0-255 range
        
        # Calculate ELA score (mean difference)
        ela_score = np.mean(diff)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return ela_score
        
    except Exception as e:
        print(f"ELA Analysis Error: {str(e)}", file=sys.stderr)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return 0.0

def is_valid_pan(pan_number):
    """Check if PAN number matches the required pattern"""
    return bool(PAN_PATTERN.match(pan_number))

def enhance_image_for_ocr(image):
    """Apply advanced image preprocessing for better OCR"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Noise removal
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        # Deskewing
        coords = np.column_stack(np.where(denoised > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            denoised, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    except Exception as e:
        print(f"Image enhancement error: {str(e)}")
        return image

def extract_pan_details(image_path):
    """Extract all relevant details from PAN card"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError("Failed to load image")
            
        # Enhance image
        processed = enhance_image_for_ocr(image)
        results = {'confidence': 0.0}
        
        # Extract text from specific zones
        for zone_name, (x1, y1, x2, y2) in PAN_ZONES.items():
            zone_image = processed[y1:y2, x1:x2]
            text = pytesseract.image_to_string(
                zone_image,
                config='--oem 3 --psm 6'
            ).strip()
            results[f'extracted_{zone_name}'] = text
            
        # Look for PAN pattern
        full_text = pytesseract.image_to_string(processed)
        pan_matches = PAN_PATTERN.findall(full_text)
        results['pan_number'] = pan_matches[0] if pan_matches else None
        
        # Detect face using Haar cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(processed, 1.1, 4)
        results['face_detected'] = len(faces) > 0
        
        # Calculate confidence score based on multiple factors
        confidence_factors = [
            1.0 if results['pan_number'] else 0.0,
            0.3 if results['face_detected'] else 0.0,
            0.2 if results.get('extracted_signature') else 0.0,
            0.2 if results.get('extracted_name') else 0.0,
            0.3 if results.get('extracted_dob') else 0.0
        ]
        results['confidence'] = sum(confidence_factors) / len(confidence_factors)
        
        # Detect potential manipulation
        ela_score = perform_ela(image_path)
        results['manipulated_score'] = ela_score / 100.0
        
        return results
        
    except Exception as e:
        print(f"Error extracting PAN details: {str(e)}")
        return {'error': str(e), 'confidence': 0.0}

def extract_pan_from_image(image_path):
    """Extract text from PAN card image using OCR"""
    try:
        # Verify tesseract installation
        if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            raise RuntimeError(
                f"Tesseract not found at: {pytesseract.pytesseract.tesseract_cmd}\n"
                "Please verify installation and update the path in the code."
            )
            
        # Read and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")
            
        # Image preprocessing
        # Resize if too small
        scale_percent = 200
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to preprocess the image
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Apply dilation to connect text components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        gray = cv2.dilate(gray, kernel, iterations=1)
        
        # Write the grayscale image to disk as temporary file
        temp_file = os.path.join(app.config['UPLOAD_FOLDER'], "temp_ocr.png")
        cv2.imwrite(temp_file, gray)
        
        try:
            # Perform OCR
            text = pytesseract.image_to_string(
                Image.open(temp_file),
                config='--oem 3 --psm 6'
            )
            
            # Clean up temporary file
            os.remove(temp_file)
            
            # Look for PAN pattern in extracted text
            pan_matches = PAN_PATTERN.findall(text)
            print(f"Extracted text: {text}")
            print(f"Found PAN matches: {pan_matches}")
            
            return {
                'success': True,
                'text': text,
                'pan_number': pan_matches[0] if pan_matches else None,
                'confidence': 0.85 if pan_matches else 0.0
            }
            
        except Exception as ocr_error:
            raise RuntimeError(f"OCR failed: {str(ocr_error)}")
            
    except Exception as e:
        print(f"Error in extract_pan_from_image: {str(e)}", file=sys.stderr)
        return {
            'success': False,
            'error': str(e),
            'text': None,
            'pan_number': None,
            'confidence': 0.0
        }

def extract_pan_from_pdf(pdf_path):
    """Extract text from PAN card PDF"""
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            pan_matches = PAN_PATTERN.findall(text)
            return {
                'success': True,
                'text': text,
                'pan_number': pan_matches[0] if pan_matches else None,
                'confidence': 0.85 if pan_matches else 0.0
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'text': None,
            'pan_number': None,
            'confidence': 0.0
        }

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)