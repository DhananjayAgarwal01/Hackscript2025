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
import fitz  # PyMuPDF
import hashlib
from pathlib import Path

warnings.filterwarnings('ignore')

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

app = Flask(__name__, template_folder='templates')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agriprice.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
db = SQLAlchemy(app)

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
                analysis_result=str(analysis_results),
                metadata=str(analysis_results.get('metadata_analysis', {}))
            )
            db.session.add(new_document)
            db.session.commit()
            
            # Save detailed analysis results
            for analysis_type, result in analysis_results.items():
                if isinstance(result, dict):
                    analysis_entry = AnalysisResult(
                        document_id=new_document.id,
                        analysis_type=analysis_type,
                        result=str(result),
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
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
    
    # Basic content analysis
    results['content_analysis']['text_length'] = len(text)
    results['content_analysis']['has_suspicious_patterns'] = False
    
    # Structural analysis for images
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = cv2.imread(file_path)
        # Error Level Analysis
        ela_score = perform_ela(file_path)
        results['structural_analysis']['ela_score'] = ela_score
    
    # Metadata analysis
    if file_path.lower().endswith('.pdf'):
        doc = fitz.open(file_path)
        metadata = doc.metadata
        results['metadata_analysis'] = metadata
    
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
    QUALITY = 90
    temp_path = "temp_ela.jpg"
    
    original = Image.open(image_path)
    original.save(temp_path, 'JPEG', quality=QUALITY)
    temporary = Image.open(temp_path)
    
    diff = Image.open(image_path).convert("L") - Image.open(temp_path).convert("L")
    diff = diff.point(lambda p: 255 if p > 0 else 0, '1')
    diff = diff.convert("RGB")
    
    # Calculate the average difference
    sum_diff = 0
    for x in range(diff.width):
        for y in range(diff.height):
            r, g, b = diff.getpixel((x, y))
            sum_diff += (r + g + b) / 3
    
    os.remove(temp_path)
    return (sum_diff / (diff.width * diff.height)) * 100

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)