from flask import Flask, render_template, request, flash, redirect, jsonify, url_for, send_file
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
import qrcode
from io import BytesIO
import base64
from skimage.metrics import structural_similarity as ssim
import imutils

warnings.filterwarnings('ignore')

def initialize_tesseract():
    """Initialize Tesseract with proper path detection"""
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\DELL\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract',
    ]
    
    tesseract_found = False
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"Found Tesseract at: {path}")
            tesseract_found = True
            break
    
    if not tesseract_found:
        error_message = """
        Tesseract OCR is not installed or not found in common locations.
        Please install Tesseract OCR:
        1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
        2. Install and note the installation path
        3. Make sure to add Tesseract to your system PATH during installation
        4. Restart your application after installation
        """
        print(error_message)
        return False
    
    try:
        # Test if Tesseract is working
        test_image = np.zeros((50, 200), dtype=np.uint8)
        cv2.putText(test_image, "Test", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        result = pytesseract.image_to_string(test_image)
        print("Tesseract OCR test result:", result)
        print("Tesseract OCR is working correctly")
        return True
    except Exception as e:
        print(f"Error testing Tesseract: {str(e)}")
        return False

UPLOAD_FOLDER = os.path.join('static', 'uploads')
PROCESSED_FOLDER = os.path.join('static', 'processed')
TEMPLATES_FOLDER = os.path.join('static', 'templates')
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
PAN_PATTERN = re.compile(r'^[A-Z]{5}[0-9]{4}[A-Z]$')
AADHAAR_PATTERN = re.compile(r'^\d{4}\s?\d{4}\s?\d{4}$')

def setup_directories():
    """Create necessary directories with proper permissions"""
    try:
        # Create base static directory if it doesn't exist
        if not os.path.exists('static'):
            os.makedirs('static')
            print("Created static directory")

        # Create required subdirectories
        directories = [
            os.path.join('static', 'uploads'),
            os.path.join('static', 'processed'),
            os.path.join('static', 'templates')
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            else:
                print(f"Directory already exists: {directory}")

            # Ensure directory is writable
            try:
                test_file = os.path.join(directory, 'test.txt')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                print(f"Directory {directory} is writable")
            except Exception as e:
                print(f"Warning: Directory {directory} may not be writable: {str(e)}")
                
    except Exception as e:
        print(f"Error setting up directories: {str(e)}")
        raise

app = Flask(__name__, template_folder='templates')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///documents.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
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

# Initialize Tesseract
tesseract_initialized = False
try:
    print("\nInitializing Tesseract OCR...")
    tesseract_initialized = initialize_tesseract()
    if tesseract_initialized:
        print("Tesseract OCR initialized successfully")
    else:
        print("Warning: Document verification features will not work without Tesseract OCR")
except Exception as e:
    print(f"Error initializing Tesseract: {str(e)}")
    print("Warning: Document verification features will not work without Tesseract OCR")

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
    processed_image = db.Column(db.String(255))

class AadhaarVerification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    aadhaar_number = db.Column(db.String(12), nullable=False)
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    status = db.Column(db.String(20), nullable=False)
    confidence_score = db.Column(db.Float)
    name = db.Column(db.String(100))
    dob = db.Column(db.String(20))
    gender = db.Column(db.String(10))
    address = db.Column(db.Text)
    qr_verified = db.Column(db.Boolean, default=False)
    biometric_status = db.Column(db.String(50))
    security_features = db.Column(db.Text)
    last_updated = db.Column(db.DateTime)
    filename = db.Column(db.String(255))
    processed_image = db.Column(db.String(255))
    manipulated_score = db.Column(db.Float, default=0.0)

class ChequeVerification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    verification_status = db.Column(db.String(20), nullable=False)
    confidence_score = db.Column(db.Float)
    extracted_text = db.Column(db.Text)
    filename = db.Column(db.String(255))
    manipulated_score = db.Column(db.Float, default=0.0)
    processed_image = db.Column(db.String(255))
    bank_name = db.Column(db.String(100))
    cheque_number = db.Column(db.String(50))
    amount = db.Column(db.String(50))
    payee_name = db.Column(db.String(100))
    date = db.Column(db.String(20))

class PassportVerification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    verification_status = db.Column(db.String(20), nullable=False)
    confidence_score = db.Column(db.Float)
    filename = db.Column(db.String(255))
    processed_image = db.Column(db.String(255))
    difference_image = db.Column(db.String(255))
    manipulated_score = db.Column(db.Float, default=0.0)

class BankStatementVerification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    upload_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    verification_status = db.Column(db.String(20), nullable=False)
    real_score = db.Column(db.Float)
    fake_score = db.Column(db.Float)
    filename = db.Column(db.String(255))
    processed_image = db.Column(db.String(255))

# ********************************** Main Routes **********************************
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'GET':
        return render_template('contact.html')
    
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

# ********************************** Document Verification Routes **********************************
@app.route('/verify')
def verify():
    return render_template('verify_document.html')

@app.route('/verify-pan', methods=['GET'])
def verify_pan_form():
    return render_template('verify_pan.html')

@app.route('/verify-pan', methods=['POST'])
def verify_pan():
    try:
        print("\nStarting PAN verification process...")
        print("Request method:", request.method)
        print("Request files:", request.files)
        print("Form data:", request.form)
        
        # Check if Tesseract is initialized
        if not tesseract_initialized:
            print("Tesseract not initialized")
            flash('OCR system is not initialized. Please ensure Tesseract is installed.', 'error')
            return redirect(url_for('verify_pan_form'))
        
        if 'pan_file' not in request.files:
            print("No file part in request")
            flash('No file uploaded', 'error')
            return redirect(url_for('verify_pan_form'))

        file = request.files['pan_file']
        print(f"Received file: {file.filename}")

        if file.filename == '':
            print("No selected file")
            flash('No file selected', 'error')
            return redirect(url_for('verify_pan_form'))

        try:
            # Save the uploaded file
            print("Attempting to save file...")
            file_path, filename = save_uploaded_file(file, prefix='pan')
            if not file_path:
                print("Failed to save file")
                flash('Invalid file type or upload failed', 'error')
                return redirect(url_for('verify_pan_form'))
            print(f"File saved successfully at: {file_path}")

            print("Starting PAN card verification...")
            result = verify_pan_card(file_path)
            
            if result is None:
                print("Verification returned None result")
                flash('Error processing the document. Please ensure the image is clear and try again.', 'error')
                cleanup_file(file_path)
                return redirect(url_for('verify_pan_form'))

            print("Verification completed. Processing results...")
            print("Result:", result)

            # Calculate confidence percentage
            confidence_percentage = result['confidence_score'] * 100

            # Determine verification status based on confidence score
            if confidence_percentage > 70:
                verification_status = 'AUTHENTIC'
            elif 50 <= confidence_percentage <= 70:
                verification_status = 'SUSPICIOUS'
            else:
                verification_status = 'FORGED'

            print(f"Status: {verification_status}, Confidence: {confidence_percentage}%")

            try:
                print("Creating verification record...")
                verification = PanVerification(
                    pan_number=result.get('pan_number', ''),
                    verification_status=verification_status,
                    confidence_score=result['confidence_score'],
                    extracted_text=json.dumps(result),
                    filename=filename,
                    extracted_name=result['extracted_name'],
                    extracted_dob=result['extracted_dob'],
                    face_detected=result['face_detected'],
                    manipulated_score=result['manipulated_score'],
                    processed_image=result['processed_image']
                )
                db.session.add(verification)
                db.session.commit()
                print("Verification record saved to database")

                # Clean up the original uploaded file
                cleanup_file(file_path)

                print("Rendering result template...")
                return render_template('pan_result.html',
                                    verification=verification,
                                    status=verification_status,
                                    confidence_score=confidence_percentage,
                                    extracted_name=result['extracted_name'],
                                    extracted_dob=result['extracted_dob'],
                                    face_detected=result['face_detected'],
                                    manipulated_score=result['manipulated_score'] * 100,
                                    processed_image=result['processed_image'])

            except Exception as e:
                print(f"Database error: {str(e)}")
                db.session.rollback()
                flash('Error saving verification results', 'error')
                cleanup_file(file_path)
                return redirect(url_for('verify_pan_form'))

        except Exception as e:
            print(f"Error processing file: {str(e)}")
            import traceback
            traceback.print_exc()
            if 'file_path' in locals():
                cleanup_file(file_path)
            flash('Error processing the document', 'error')
            return redirect(url_for('verify_pan_form'))

    except Exception as e:
        print(f"Error during verification: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Error during verification: {str(e)}', 'error')
        return redirect(url_for('verify_pan_form'))

@app.route('/verify-aadhaar', methods=['GET'])
def verify_aadhaar_form():
    return render_template('verify_aadhaar.html')

@app.route('/verify-aadhaar', methods=['POST'])
def verify_aadhaar():
    try:
        print("\nStarting Aadhaar verification process...")
        print("Request method:", request.method)
        print("Request files:", request.files)
        print("Form data:", request.form)
        
        # Check if Tesseract is initialized
        if not tesseract_initialized:
            print("Tesseract not initialized")
            flash('OCR system is not initialized. Please ensure Tesseract is installed.', 'error')
            return redirect(url_for('verify_aadhaar_form'))
        
        if 'aadhaar_file' not in request.files:
            print("No file part in request")
            flash('No file uploaded', 'error')
            return redirect(url_for('verify_aadhaar_form'))

        file = request.files['aadhaar_file']
        print(f"Received file: {file.filename}")

        if file.filename == '':
            print("No selected file")
            flash('No file selected', 'error')
            return redirect(url_for('verify_aadhaar_form'))

        try:
            # Save the uploaded file
            print("Attempting to save file...")
            file_path, filename = save_uploaded_file(file, prefix='aadhaar')
            if not file_path:
                print("Failed to save file")
                flash('Invalid file type or upload failed', 'error')
                return redirect(url_for('verify_aadhaar_form'))
            print(f"File saved successfully at: {file_path}")

            print("Starting Aadhaar card verification...")
            result = verify_aadhaar_card(file_path)
            
            if result is None:
                print("Verification returned None result")
                flash('Error processing the document. Please ensure the image is clear and try again.', 'error')
                cleanup_file(file_path)
                return redirect(url_for('verify_aadhaar_form'))

            print("Verification completed. Processing results...")
            print("Result:", result)

            # Calculate confidence percentage
            confidence_percentage = result['confidence_score'] * 100

            # Determine verification status based on confidence score
            if confidence_percentage > 70:
                status = 'AUTHENTIC'
            elif 50 <= confidence_percentage <= 70:
                status = 'SUSPICIOUS'
            else:
                status = 'FORGED'

            print(f"Status: {status}, Confidence: {confidence_percentage}%")

            try:
                print("Creating verification record...")
                verification = AadhaarVerification(
                    aadhaar_number=result.get('aadhaar_number', ''),
                    status=status,
                    confidence_score=result['confidence_score'],
                    name=result['name'],
                    dob=result['dob'],
                    gender=result['gender'],
                    address=result['address'],
                    qr_verified=result['qr_verified'],
                    biometric_status=result['biometric_status'],
                    security_features=result['security_features'],
                    last_updated=result['last_updated'],
                    filename=filename,
                    processed_image=result['processed_image'],
                    manipulated_score=result['manipulated_score']
                )

                db.session.add(verification)
                db.session.commit()
                print("Verification record saved to database")

                # Clean up the original uploaded file
                cleanup_file(file_path)

                print("Rendering result template...")
                return render_template('aadhaar_result.html',
                                    verification=verification,
                                    status=status,
                                    confidence_score=confidence_percentage,
                                    name=result['name'],
                                    dob=result['dob'],
                                    gender=result['gender'],
                                    address=result['address'],
                                    qr_verified=result['qr_verified'],
                                    biometric_status=result['biometric_status'],
                                    security_features=result['security_features'],
                                    processed_image=result['processed_image'],
                                    manipulated_score=result['manipulated_score'] * 100)

            except Exception as e:
                print(f"Database error: {str(e)}")
                db.session.rollback()
                flash('Error saving verification results', 'error')
                cleanup_file(file_path)
                return redirect(url_for('verify_aadhaar_form'))

        except Exception as e:
            print(f"Error processing file: {str(e)}")
            import traceback
            traceback.print_exc()
            if 'file_path' in locals():
                cleanup_file(file_path)
            flash('Error processing the document', 'error')
            return redirect(url_for('verify_aadhaar_form'))

    except Exception as e:
        print(f"Error during verification: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Error during verification: {str(e)}', 'error')
        return redirect(url_for('verify_aadhaar_form'))

@app.route('/pan-history')
def pan_verification_history():
    verifications = PanVerification.query.order_by(PanVerification.upload_date.desc()).all()
    return render_template('pan_history.html', verifications=verifications)

@app.route('/aadhaar-history')
def aadhaar_verification_history():
    verifications = AadhaarVerification.query.order_by(AadhaarVerification.upload_date.desc()).all()
    return render_template('aadhaar_history.html', verifications=verifications)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename))

@app.route('/verify-cheque', methods=['GET'])
def verify_cheque_form():
    return render_template('verify_cheque.html')

@app.route('/verify-cheque', methods=['POST'])
def verify_cheque():
    try:
        if 'cheque_file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for('verify_cheque_form'))

        file = request.files['cheque_file']

        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('verify_cheque_form'))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            result = verify_cheque_image(file_path)
            
            if result is None:
                flash('Error processing the document', 'error')
                return redirect(url_for('verify_cheque_form'))

            # Calculate confidence percentage
            confidence_percentage = result['confidence_score'] * 100

            # Determine verification status based on confidence score
            if confidence_percentage > 70:
                verification_status = 'Real Cheque'
            elif 50 <= confidence_percentage <= 70:
                verification_status = 'Suspicious'
            else:
                verification_status = 'Fake Cheque'

            verification = ChequeVerification(
                verification_status=verification_status,
                confidence_score=result['confidence_score'],
                extracted_text=json.dumps(result['extracted_text']),
                filename=filename,
                manipulated_score=result['manipulated_score'],
                processed_image=result['processed_image'],
                bank_name=result.get('bank_name', ''),
                cheque_number=result.get('cheque_number', ''),
                amount=result.get('amount', ''),
                payee_name=result.get('payee_name', ''),
                date=result.get('date', '')
            )
            db.session.add(verification)
            db.session.commit()

            os.remove(file_path)

            return render_template('cheque_result.html',
                                verification=verification,
                                status=verification_status,
                                confidence_score=confidence_percentage,
                                bank_name=result.get('bank_name', ''),
                                cheque_number=result.get('cheque_number', ''),
                                amount=result.get('amount', ''),
                                payee_name=result.get('payee_name', ''),
                                date=result.get('date', ''),
                                manipulated_score=result['manipulated_score'] * 100,
                                processed_image=result['processed_image'])

        flash('Invalid file type', 'error')
        return redirect(url_for('verify_cheque_form'))

    except Exception as e:
        flash(f'Error during verification: {str(e)}', 'error')
        return redirect(url_for('verify_cheque_form'))

@app.route('/cheque-history')
def cheque_verification_history():
    verifications = ChequeVerification.query.order_by(ChequeVerification.upload_date.desc()).all()
    return render_template('cheque_history.html', verifications=verifications)

@app.route('/verify-passport', methods=['GET'])
def verify_passport_form():
    return render_template('verify_passport.html')

@app.route('/verify-passport', methods=['POST'])
def verify_passport_submit():
    try:
        if 'passport_file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for('verify_passport_form'))

        file = request.files['passport_file']

        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('verify_passport_form'))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            result = verify_passport(file_path)
            
            if result is None or result.get('verification_status') == 'ERROR':
                error_message = result.get('error_message', 'Error processing the document') if result else 'Error processing the document'
                flash(error_message, 'error')
                if os.path.exists(file_path):
                    os.remove(file_path)
                return redirect(url_for('verify_passport_form'))

            verification = PassportVerification(
                verification_status=result['verification_status'],
                confidence_score=result['confidence_score'],
                filename=filename,
                processed_image=result['processed_image'],
                difference_image=result['difference_image'],
                manipulated_score=result['manipulated_score']
            )
            db.session.add(verification)
            db.session.commit()

            if os.path.exists(file_path):
                os.remove(file_path)

            return render_template('passport_result.html',
                                verification=verification,
                                status=result['verification_status'],
                                confidence_score=result['confidence_score'] * 100,
                                manipulated_score=result['manipulated_score'] * 100,
                                processed_image=result['processed_image'],
                                difference_image=result['difference_image'])

        flash('Invalid file type', 'error')
        return redirect(url_for('verify_passport_form'))

    except Exception as e:
        flash(f'Error during verification: {str(e)}', 'error')
        return redirect(url_for('verify_passport_form'))

@app.route('/passport-history')
def passport_verification_history():
    verifications = PassportVerification.query.order_by(PassportVerification.upload_date.desc()).all()
    return render_template('passport_history.html', verifications=verifications)

@app.route('/verify-bank-statement', methods=['GET'])
def verify_bank_statement_form():
    return render_template('verify_bank_statement.html')

@app.route('/verify-bank-statement', methods=['POST'])
def verify_bank_statement_submit():
    try:
        if 'bank_statement_file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for('verify_bank_statement_form'))

        file = request.files['bank_statement_file']

        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('verify_bank_statement_form'))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            result = verify_bank_statement(file_path)
            
            if result is None:
                flash('Error processing the document', 'error')
                return redirect(url_for('verify_bank_statement_form'))

            verification = BankStatementVerification(
                verification_status=result['verification_status'],
                real_score=result['real_score'],
                fake_score=result['fake_score'],
                filename=filename,
                processed_image=result['processed_image']
            )
            db.session.add(verification)
            db.session.commit()

            os.remove(file_path)

            return render_template('bank_statement_result.html',
                                verification=verification,
                                status=result['verification_status'],
                                real_score=result['real_score'] * 100,
                                fake_score=result['fake_score'] * 100,
                                processed_image=result['processed_image'])

        flash('Invalid file type', 'error')
        return redirect(url_for('verify_bank_statement_form'))

    except Exception as e:
        flash(f'Error during verification: {str(e)}', 'error')
        return redirect(url_for('verify_bank_statement_form'))

@app.route('/bank-statement-history')
def bank_statement_verification_history():
    verifications = BankStatementVerification.query.order_by(BankStatementVerification.upload_date.desc()).all()
    return render_template('bank_statement_history.html', verifications=verifications)

# ********************************** Error Handlers **********************************
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

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

def verify_pan_card(image_path):
    """Verify PAN card and extract information"""
    try:
        print(f"\nStarting PAN card verification for: {image_path}")
        
        # Process image
        print("Processing image...")
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + os.path.basename(image_path))
        uploaded_gray, uploaded_img = preprocess_image_for_verification(image_path)
        if uploaded_gray is None or uploaded_img is None:
            print("Error: Failed to process image")
            return None

        # Define zones for PAN card
        height, width = uploaded_img.shape[:2]
        zones = {
            'name': (int(width*0.2), int(height*0.2), int(width*0.8), int(height*0.3)),
            'pan_number': (int(width*0.2), int(height*0.4), int(width*0.8), int(height*0.5)),
            'dob': (int(width*0.2), int(height*0.3), int(width*0.8), int(height*0.4)),
        }

        # Extract text from zones
        print("Extracting text from zones...")
        results = {}
        for zone_name, (x1, y1, x2, y2) in zones.items():
            roi = uploaded_img[y1:y2, x1:x2]
            text = pytesseract.image_to_string(roi, config="--psm 6")
            results[zone_name] = text.strip()
            cv2.rectangle(uploaded_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Detect face
        print("Detecting face...")
        faces = detect_faces_in_image(uploaded_img)
        face_detected = len(faces) > 0
        print(f"Face detected: {face_detected}")

        # Draw blue boxes for detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(uploaded_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Compare with reference template if available
        ssim_score = 0
        reference_path = os.path.join('static', 'templates', 'reference_pan.png')
        if os.path.exists(reference_path):
            print("Comparing with reference template...")
            reference_gray, _ = preprocess_image_for_verification(reference_path)
            if reference_gray is not None:
                score, diff = ssim(reference_gray, uploaded_gray, full=True)
                ssim_score = score
                print(f"SSIM Score: {ssim_score}")

                # Process difference image
                diff = (diff * 255).astype("uint8")
                blurred = cv2.GaussianBlur(diff, (5, 5), 0)
                _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                
                # Find tampering regions
                cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)

                # Draw red boxes for tampered regions
                for c in cnts:
                    (x, y, w, h) = cv2.boundingRect(c)
                    if w > 30 and h > 30:  # Ignore small noises
                        cv2.rectangle(uploaded_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Save processed image
        cv2.imwrite(processed_path, uploaded_img)
        print(f"Saved processed image to: {processed_path}")

        # Calculate base confidence score
        confidence_factors = {
            'face_detection': 0.3 if face_detected else 0.0,
            'text_extraction': 0.3 if any(results.values()) else 0.0,
            'image_similarity': 0.4 * ssim_score
        }
        base_confidence = sum(confidence_factors.values())
        
        # Apply 1.5x multiplier to confidence score
        confidence_score = min(base_confidence * 1.5, 1.0)  # Cap at 1.0
        print(f"Base confidence: {base_confidence}, Final confidence: {confidence_score}")

        # Determine manipulation score
        manipulated_score = 1 - confidence_score
        print(f"Manipulation score: {manipulated_score}")

        return {
            'pan_number': results.get('pan_number', ''),
            'extracted_name': results.get('name', ''),
            'extracted_dob': results.get('dob', ''),
            'face_detected': face_detected,
            'confidence_score': confidence_score,
            'manipulated_score': manipulated_score,
            'processed_image': os.path.basename(processed_path)
        }

    except Exception as e:
        print(f"Error in PAN verification: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def verify_aadhaar_card(image_path):
    """Verify Aadhaar card and extract information"""
    try:
        print(f"\nStarting Aadhaar card verification for: {image_path}")
        
        # Process uploaded image
        print("Processing uploaded image...")
        uploaded_gray, uploaded_img = preprocess_image_for_verification(image_path)
        if uploaded_gray is None or uploaded_img is None:
            print("Error: Failed to process uploaded image")
            return None

        # Extract text from image
        print("Extracting text from image...")
        text = pytesseract.image_to_string(uploaded_img)
        
        # Extract required fields
        name = None
        dob = None
        gender = None
        address = None
        
        # Simple extraction based on common patterns
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # Look for name (usually after "Name:" or at the beginning)
            if "Name:" in line or "नाम:" in line:
                name = line.split(":")[-1].strip()
            elif not name and len(line) > 0 and any(c.isalpha() for c in line):
                name = line
            
            # Look for DOB
            if "DOB" in line or "Date of Birth" in line or "जन्म तिथि" in line:
                dob = line.split(":")[-1].strip()
            elif re.search(r'\d{2}/\d{2}/\d{4}', line):
                dob = re.search(r'\d{2}/\d{2}/\d{4}', line).group()
            
            # Look for gender
            if "Gender" in line or "लिंग" in line:
                gender = line.split(":")[-1].strip()
            elif "MALE" in line.upper():
                gender = "MALE"
            elif "FEMALE" in line.upper():
                gender = "FEMALE"
            
            # Look for address (usually the longest line with numbers and text)
            if len(line) > 50 and any(c.isdigit() for c in line) and any(c.isalpha() for c in line):
                address = line

        # Extract Aadhaar number using OCR
        print("Extracting Aadhaar number...")
        aadhaar_number = extract_aadhaar_number_from_image(image_path)
        print(f"Extracted Aadhaar number: {aadhaar_number if aadhaar_number else 'None'}")

        # Detect faces
        print("Detecting faces...")
        faces = detect_faces_in_image(uploaded_img)
        face_detected = len(faces) > 0
        print(f"Face detected: {face_detected}")

        # Compare with reference image if available
        ssim_score = 0
        reference_path = os.path.join('static', 'templates', 'reference_aadhaar.png')
        if os.path.exists(reference_path):
            print("Comparing with reference image...")
            reference_gray, _ = preprocess_image_for_verification(reference_path)
            if reference_gray is not None:
                score, diff = ssim(reference_gray, uploaded_gray, full=True)
                ssim_score = score
                print(f"SSIM Score: {ssim_score}")

                # Process difference image
                diff = (diff * 255).astype("uint8")
                blurred = cv2.GaussianBlur(diff, (5, 5), 0)
                _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                
                # Find tampering regions
                cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)

                # Draw red boxes for tampered regions
                for c in cnts:
                    (x, y, w, h) = cv2.boundingRect(c)
                    if w > 30 and h > 30:  # Ignore small noises
                        cv2.rectangle(uploaded_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Draw blue boxes for detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(uploaded_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Save processed image
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + os.path.basename(image_path))
        cv2.imwrite(processed_path, uploaded_img)
        print(f"Saved processed image to: {processed_path}")

        # Calculate required fields score
        required_fields_score = 0
        if name and name != "Not detected":
            required_fields_score += 0.25
        if dob and dob != "Not detected":
            required_fields_score += 0.25
        if gender and gender != "Not detected":
            required_fields_score += 0.25
        if address and address != "Not detected":
            required_fields_score += 0.25

        # Calculate base confidence score
        confidence_factors = {
            'face_detection': 0.2 if face_detected else 0.0,
            'aadhaar_number': 0.2 if aadhaar_number else 0.0,
            'image_similarity': 0.2 * ssim_score,
            'required_fields': 0.4 * required_fields_score
        }
        
        base_confidence = sum(confidence_factors.values())
        print("Confidence factors:", confidence_factors)
        
        # Apply 1.5x multiplier to confidence score
        confidence_score = min(base_confidence * 1.5, 1.0)  # Cap at 1.0
        print(f"Base confidence: {base_confidence}, Final confidence: {confidence_score}")

        # Determine manipulation score
        manipulated_score = 1 - confidence_score
        print(f"Manipulation score: {manipulated_score}")

        # Prepare result
        result = {
            'aadhaar_number': aadhaar_number if aadhaar_number else '',
            'confidence_score': confidence_score,
            'manipulated_score': manipulated_score,
            'name': name if name else 'Not detected',
            'dob': dob if dob else 'Not detected',
            'gender': gender if gender else 'Not detected',
            'address': address if address else 'Not detected',
            'qr_verified': False,
            'biometric_status': 'Not Available',
            'security_features': 'Standard Verification',
            'last_updated': datetime.utcnow(),
            'processed_image': os.path.basename(processed_path)
        }

        print("Verification completed successfully")
        return result

    except Exception as e:
        print(f"Error in Aadhaar verification: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def preprocess_image_for_verification(image_path):
    """Preprocess image for verification"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image from {image_path}")
            return None, None

        # Resize image
        image = cv2.resize(image, (600, 400))
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return gray, image

    except Exception as e:
        print(f"Error in image preprocessing: {str(e)}")
        return None, None

def extract_aadhaar_number_from_image(image_path):
    """Extract Aadhaar number using OCR"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Perform OCR
        text = pytesseract.image_to_string(gray, config="--psm 6")
        
        # Extract Aadhaar numbers (12-digit sequences)
        aadhaar_numbers = [num.replace(" ", "") for num in text.split("\n") 
                          if len(num.replace(" ", "")) == 12 and num.replace(" ", "").isdigit()]

        return aadhaar_numbers[0] if aadhaar_numbers else None

    except Exception as e:
        print(f"Error extracting Aadhaar number: {str(e)}")
        return None

def detect_faces_in_image(image):
    """Detect faces in an image"""
    try:
        # Initialize face cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        return faces

    except Exception as e:
        print(f"Error detecting faces: {str(e)}")
        return []

def verify_cheque_image(image_path):
    """Verify cheque and extract information"""
    try:
        # Process image
        processed_path = os.path.join(
            app.config['PROCESSED_FOLDER'],
            'processed_' + os.path.basename(image_path)
        )
        if not preprocess_image(image_path, processed_path):
            return None

        # Read processed image
        image = cv2.imread(processed_path)
        
        # Define zones for cheque
        height, width = image.shape[:2]
        zones = {
            'bank_name': (int(width*0.05), int(height*0.05), int(width*0.4), int(height*0.15)),
            'payee_name': (int(width*0.1), int(height*0.3), int(width*0.6), int(height*0.4)),
            'amount': (int(width*0.6), int(height*0.3), int(width*0.9), int(height*0.4)),
            'date': (int(width*0.6), int(height*0.1), int(width*0.9), int(height*0.2)),
            'cheque_number': (int(width*0.6), int(height*0.8), int(width*0.9), int(height*0.9))
        }

        # Extract text from zones
        results = extract_text_from_zones(image, zones)

        # Calculate manipulation score using ELA
        ela_score = perform_ela(image_path)
        manipulated_score = ela_score / 100.0

        # Calculate confidence score based on multiple factors
        confidence_factors = {
            'text_extraction': 0.6 if any(results.values()) else 0.0,
            'manipulation': 0.4 * (1 - manipulated_score)
        }
        
        confidence_score = sum(confidence_factors.values())

        # Draw rectangles around detected regions
        for zone_name, (x1, y1, x2, y2) in zones.items():
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Save annotated image
        annotated_path = os.path.join(
            app.config['PROCESSED_FOLDER'],
            'annotated_' + os.path.basename(image_path)
        )
        cv2.imwrite(annotated_path, image)

        return {
            'bank_name': results.get('bank_name', ''),
            'cheque_number': results.get('cheque_number', ''),
            'amount': results.get('amount', ''),
            'payee_name': results.get('payee_name', ''),
            'date': results.get('date', ''),
            'confidence_score': confidence_score,
            'manipulated_score': manipulated_score,
            'processed_image': os.path.basename(annotated_path),
            'extracted_text': results
        }

    except Exception as e:
        print(f"Error in cheque verification: {str(e)}")
        return None

def verify_passport(image_path):
    """Verify passport authenticity"""
    try:
        # Define template path
        template_path = os.path.join('static', 'templates', 'passport_template.jpg')
        
        # Check if template exists
        if not os.path.exists(template_path):
            print(f"Template file not found at {template_path}")
            return {
                'confidence_score': 0.0,
                'verification_status': 'ERROR',
                'processed_image': None,
                'difference_image': None,
                'manipulated_score': 1.0,
                'error_message': 'Template file not found'
            }

        # Read original passport template
        original_image = cv2.imread(template_path)
        if original_image is None:
            print(f"Failed to read template image at {template_path}")
            return {
                'confidence_score': 0.0,
                'verification_status': 'ERROR',
                'processed_image': None,
                'difference_image': None,
                'manipulated_score': 1.0,
                'error_message': 'Failed to read template image'
            }

        # Read uploaded image
        uploaded_image = cv2.imread(image_path)
        if uploaded_image is None:
            print(f"Failed to read uploaded image at {image_path}")
            return {
                'confidence_score': 0.0,
                'verification_status': 'ERROR',
                'processed_image': None,
                'difference_image': None,
                'manipulated_score': 1.0,
                'error_message': 'Failed to read uploaded image'
            }

        # Resize images to ensure they have the same dimensions
        uploaded_image_resized = cv2.resize(uploaded_image, (original_image.shape[1], original_image.shape[0]))

        # Convert both images to grayscale
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        uploaded_gray = cv2.cvtColor(uploaded_image_resized, cv2.COLOR_BGR2GRAY)

        # Compute SSIM
        score, diff = ssim(original_gray, uploaded_gray, full=True)
        print(f"SSIM Score: {score}")  # Debug print

        # Create difference image
        diff = (diff * 255).astype("uint8")
        diff_image_path = os.path.join(
            app.config['PROCESSED_FOLDER'],
            'diff_' + os.path.basename(image_path)
        )
        cv2.imwrite(diff_image_path, diff)

        # Save processed image
        processed_image_path = os.path.join(
            app.config['PROCESSED_FOLDER'],
            'processed_' + os.path.basename(image_path)
        )
        cv2.imwrite(processed_image_path, uploaded_image_resized)

        # Determine verification status
        if score > 0.90:
            status = 'AUTHENTIC'
        elif score > 0.70:
            status = 'SUSPICIOUS'
        else:
            status = 'FORGED'

        return {
            'confidence_score': score,
            'verification_status': status,
            'processed_image': os.path.basename(processed_image_path),
            'difference_image': os.path.basename(diff_image_path),
            'manipulated_score': 1 - score
        }

    except Exception as e:
        print(f"Error in passport verification: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'confidence_score': 0.0,
            'verification_status': 'ERROR',
            'processed_image': None,
            'difference_image': None,
            'manipulated_score': 1.0,
            'error_message': str(e)
        }

def verify_bank_statement(image_path):
    """Verify bank statement authenticity"""
    try:
        # Define paths for real and fake templates
        real_templates = [
            os.path.join(app.config['UPLOAD_FOLDER'], 'bank_template1.jpg'),
            os.path.join(app.config['UPLOAD_FOLDER'], 'bank_template2.jpg')
        ]
        fake_templates = [
            os.path.join(app.config['UPLOAD_FOLDER'], 'fake_template1.jpg'),
            os.path.join(app.config['UPLOAD_FOLDER'], 'fake_template2.jpg')
        ]

        # Process uploaded image
        uploaded_gray, uploaded_img = preprocess_image(image_path)

        real_scores = []
        fake_scores = []

        # Compare with real templates
        for template_path in real_templates:
            if os.path.exists(template_path):
                reference_gray, _ = preprocess_image(template_path)
                score, _ = ssim(reference_gray, uploaded_gray, full=True)
                real_scores.append(score)

        # Compare with fake templates
        for template_path in fake_templates:
            if os.path.exists(template_path):
                reference_gray, _ = preprocess_image(template_path)
                score, _ = ssim(reference_gray, uploaded_gray, full=True)
                fake_scores.append(score)

        # Calculate scores
        max_real_score = max(real_scores) if real_scores else 0
        max_fake_score = max(fake_scores) if fake_scores else 0

        # Save processed image
        processed_image_path = os.path.join(
            app.config['PROCESSED_FOLDER'],
            'processed_' + os.path.basename(image_path)
        )
        cv2.imwrite(processed_image_path, uploaded_img)

        return {
            'real_score': max_real_score,
            'fake_score': max_fake_score,
            'verification_status': 'AUTHENTIC' if max_real_score > max_fake_score and max_real_score > 0.80 else 'FORGED',
            'processed_image': os.path.basename(processed_image_path)
        }

    except Exception as e:
            print(f"Error in bank statement verification: {str(e)}")
            return None

def save_uploaded_file(file, prefix=''):
    """Save uploaded file with proper error handling and unique filename generation"""
    try:
        if file and allowed_file(file.filename):
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            original_filename = secure_filename(file.filename)
            filename = f"{prefix}_{timestamp}_{original_filename}" if prefix else f"{timestamp}_{original_filename}"
            
            # Ensure upload directory exists
            upload_dir = os.path.join('static', 'uploads')
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
                print(f"Created upload directory: {upload_dir}")
            
            # Create full file path
            file_path = os.path.join(upload_dir, filename)
            
            # Save the file
            try:
                file.save(file_path)
                print(f"File saved successfully at: {file_path}")
                
                # Verify file was saved
                if os.path.exists(file_path):
                    print(f"File exists at: {file_path}")
                    print(f"File size: {os.path.getsize(file_path)} bytes")
                else:
                    print(f"Warning: File does not exist at: {file_path}")
                    return None, None
                
                return file_path, filename
            except Exception as e:
                print(f"Error saving file: {str(e)}")
                return None, None
        return None, None
    except Exception as e:
        print(f"Error in save_uploaded_file: {str(e)}")
        return None, None

def cleanup_file(file_path):
    """Safely remove a file if it exists"""
    try:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Successfully cleaned up file: {file_path}")
                return True
            except Exception as e:
                print(f"Error removing file {file_path}: {str(e)}")
                return False
        else:
            print(f"File not found for cleanup: {file_path}")
            return False
    except Exception as e:
        print(f"Error in cleanup_file: {str(e)}")
        return False

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)