from flask import Flask, render_template, request, jsonify, send_from_directory
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import pytesseract
import os
import imutils

app = Flask(__name__)

# Configure upload folders
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configure face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Reference image paths
REFERENCE_PATHS = {
    'aadhaar': 'reference_aadhaar.png',
    'passport': '1.jpg',
    'cheque': 'cheque_og.png',
    'bank_statement_real': ['1.jpg', '3.jpg'],
    'bank_statement_fake': ['Fake.png', 'f2.webp']
}

def verify_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reference image not found: {file_path}")

# Verify reference files exist
for key, path in REFERENCE_PATHS.items():
    if isinstance(path, list):
        for p in path:
            verify_file_exists(p)
    else:
        verify_file_exists(path)

def preprocess_image(image_path, size=(600, 400)):
    """Resize and convert image to grayscale for SSIM."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image = cv2.resize(image, size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray, image

def detect_tampered_regions(diff):
    """Detect tampered regions using contours."""
    blurred = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts, thresh

def extract_aadhaar_number(image_path):
    """Extract Aadhaar number using OCR."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(gray, config="--psm 6")
    aadhaar_numbers = [num.replace(" ", "") for num in text.split("\n") 
                      if len(num.replace(" ", "")) == 12 and num.replace(" ", "").isdigit()]
    return aadhaar_numbers[0] if aadhaar_numbers else None

def detect_faces(image):
    """Detect faces in an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    return faces

def verify_aadhaar(file_path):
    """Verify Aadhaar card."""
    uploaded_gray, uploaded_img = preprocess_image(file_path)
    reference_gray, reference_img = preprocess_image(REFERENCE_PATHS['aadhaar'])
    
    # Extract Aadhaar numbers
    uploaded_aadhaar = extract_aadhaar_number(file_path)
    reference_aadhaar = extract_aadhaar_number(REFERENCE_PATHS['aadhaar'])
    
    # Compute SSIM
    score, diff = ssim(reference_gray, uploaded_gray, full=True)
    diff = (diff * 255).astype("uint8")
    
    # Detect tampering
    cnts, thresh = detect_tampered_regions(diff)
    
    # Mark tampered regions
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 30 and h > 30:
            cv2.rectangle(uploaded_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Detect and mark face
    faces = detect_faces(uploaded_img)
    for (x, y, w, h) in faces:
        cv2.rectangle(uploaded_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Save processed images
    output_path = os.path.join(STATIC_FOLDER, 'processed_aadhaar.png')
    cv2.imwrite(output_path, uploaded_img)
    
    return {
        'ssim_score': round(score * 100, 2),
        'aadhaar_match': uploaded_aadhaar == reference_aadhaar,
        'authenticity': "Real" if score > 0.80 else "Fake",
        'processed_image': output_path
    }

def verify_passport(file_path):
    """Verify passport."""
    uploaded_image = cv2.imread(file_path)
    original_image = cv2.imread(REFERENCE_PATHS['passport'])
    
    # Resize images
    min_dimension = min(uploaded_image.shape[0], uploaded_image.shape[1],
                       original_image.shape[0], original_image.shape[1])
    if min_dimension < 7:
        uploaded_image = cv2.resize(uploaded_image, (300, 300))
        original_image = cv2.resize(original_image, (300, 300))
    
    uploaded_image_resized = cv2.resize(uploaded_image, 
                                      (original_image.shape[1], original_image.shape[0]))
    
    # Compute SSIM
    score, _ = ssim(original_image, uploaded_image_resized, full=True, win_size=3)
    
    # Create difference image
    diff_image = cv2.absdiff(original_image, uploaded_image_resized)
    diff_path = os.path.join(STATIC_FOLDER, 'passport_difference.jpg')
    cv2.imwrite(diff_path, diff_image)
    
    processed_path = os.path.join(STATIC_FOLDER, 'processed_passport.jpg')
    cv2.imwrite(processed_path, uploaded_image_resized)
    
    return {
        'ssim_score': round(score * 100, 2),
        'authenticity': "Real" if score > 0.90 else "Fake",
        'processed_image': processed_path,
        'difference_image': diff_path
    }

def verify_cheque(file_path):
    """Verify cheque."""
    uploaded_gray, uploaded_img = preprocess_image(file_path)
    reference_gray, reference_img = preprocess_image(REFERENCE_PATHS['cheque'])
    
    # Compute SSIM
    score, diff = ssim(reference_gray, uploaded_gray, full=True)
    diff = (diff * 255).astype("uint8")
    
    # Detect tampering
    cnts, thresh = detect_tampered_regions(diff)
    
    # Mark tampered regions
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 30 and h > 30:
            cv2.rectangle(uploaded_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Extract text
    text = extract_aadhaar_number(file_path)  # Reusing function for text extraction
    
    # Save processed images
    output_paths = {
        'processed': os.path.join(STATIC_FOLDER, 'processed_cheque.png'),
        'difference': os.path.join(STATIC_FOLDER, 'cheque_difference.png'),
        'threshold': os.path.join(STATIC_FOLDER, 'cheque_threshold.png')
    }
    
    cv2.imwrite(output_paths['processed'], uploaded_img)
    cv2.imwrite(output_paths['difference'], diff)
    cv2.imwrite(output_paths['threshold'], thresh)
    
    return {
        'ssim_score': round(score * 100, 2),
        'authenticity': "Real" if score > 0.80 else "Fake",
        'extracted_text': text,
        'processed_image': output_paths['processed'],
        'difference_image': output_paths['difference'],
        'threshold_image': output_paths['threshold']
    }

def verify_bank_statement(file_path):
    """Verify bank statement."""
    uploaded_gray, uploaded_img = preprocess_image(file_path)
    
    real_scores = []
    fake_scores = []
    
    # Compare with real images
    for real_image_path in REFERENCE_PATHS['bank_statement_real']:
        reference_gray, _ = preprocess_image(real_image_path)
        score, _ = ssim(reference_gray, uploaded_gray, full=True)
        real_scores.append(score)
    
    # Compare with fake images
    for fake_image_path in REFERENCE_PATHS['bank_statement_fake']:
        fake_gray, _ = preprocess_image(fake_image_path)
        score, _ = ssim(fake_gray, uploaded_gray, full=True)
        fake_scores.append(score)
    
    max_real_score = max(real_scores)
    max_fake_score = max(fake_scores)
    
    # Save processed image
    output_path = os.path.join(STATIC_FOLDER, 'processed_statement.png')
    cv2.imwrite(output_path, uploaded_img)
    
    return {
        'real_score': round(max_real_score * 100, 2),
        'fake_score': round(max_fake_score * 100, 2),
        'authenticity': "Real" if max_real_score >= max_fake_score and max_real_score > 0.80 else "Fake",
        'processed_image': output_path
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<filename>')
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        doc_type = request.form.get('doc_type', '')
        if not doc_type:
            return jsonify({'error': 'Document type not specified'}), 400
        
        # Save uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        # Process based on document type
        if doc_type == 'aadhaar':
            result = verify_aadhaar(file_path)
        elif doc_type == 'passport':
            result = verify_passport(file_path)
        elif doc_type == 'cheque':
            result = verify_cheque(file_path)
        elif doc_type == 'bank_statement':
            result = verify_bank_statement(file_path)
        else:
            return jsonify({'error': 'Invalid document type'}), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 