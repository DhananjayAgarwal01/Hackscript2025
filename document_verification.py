from flask import Flask, render_template, request, jsonify, send_from_directory
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import pytesseract
import os
import imutils

app = Flask(__name__)

# Configure upload folders
UPLOAD_FOLDER = {
    'aadhaar': os.path.join('uploads', 'aadhaar'),
    'cheque': os.path.join('uploads', 'cheque')
}

# Create upload directories if they don't exist
for folder in UPLOAD_FOLDER.values():
    os.makedirs(folder, exist_ok=True)

# Configure Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load Haar cascade for face detection (used in Aadhaar verification)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Reference images
REFERENCE_IMAGES = {
    'aadhaar': "reference_aadhaar.png",
    'cheque': "cheque_og.png"
}

def preprocess_image(image_path):
    """Resize and convert image to grayscale for SSIM."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (600, 400))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray, image

def detect_tampered_regions(diff):
    """Detect tampered regions using contours."""
    blurred = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts, thresh

def extract_text(image_path, doc_type='aadhaar'):
    """Extract text using OCR with configuration based on document type."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    if doc_type == 'aadhaar':
        # Configuration specific to Aadhaar number extraction
        text = pytesseract.image_to_string(gray, config="--psm 6")
        numbers = [num.replace(" ", "") for num in text.split("\n") 
                  if len(num.replace(" ", "")) == 12 and num.replace(" ", "").isdigit()]
        return numbers[0] if numbers else None
    else:
        # General text extraction for cheques
        return pytesseract.image_to_string(gray, config="--psm 6")

def detect_faces(image):
    """Detect faces in an image using Haar cascade."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    return faces

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<folder>/<filename>')
def uploaded_file(folder, filename):
    return send_from_directory(os.path.join(UPLOAD_FOLDER[folder]), filename)

@app.route('/verify/<doc_type>', methods=['POST'])
def verify_document(doc_type):
    if doc_type not in ['aadhaar', 'cheque']:
        return jsonify({'error': 'Invalid document type'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER[doc_type], file.filename)
    file.save(file_path)

    # Process the uploaded image
    uploaded_gray, uploaded_img = preprocess_image(file_path)
    
    # Get reference image
    reference_path = REFERENCE_IMAGES[doc_type]
    if not os.path.exists(reference_path):
        return jsonify({'error': f'Reference image not found: {reference_path}'}), 500
    
    reference_gray, reference_img = preprocess_image(reference_path)

    # Compute SSIM
    score, diff = ssim(reference_gray, uploaded_gray, full=True)
    diff = (diff * 255).astype("uint8")

    # Detect tampered regions
    cnts, thresh = detect_tampered_regions(diff)

    # Draw boxes around tampered regions
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > 30 and h > 30:  # Ignore small noise
            cv2.rectangle(uploaded_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Document-specific processing
    if doc_type == 'aadhaar':
        # Extract Aadhaar number
        uploaded_number = extract_text(file_path, 'aadhaar')
        reference_number = extract_text(reference_path, 'aadhaar')
        
        # Detect and highlight face
        faces = detect_faces(uploaded_img)
        for (x, y, w, h) in faces:
            cv2.rectangle(uploaded_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        number_match = uploaded_number == reference_number if uploaded_number and reference_number else False
        authenticity = "Original" if number_match and score >= 0.80 else "Tampered"
        
        result = {
            "ssim_score": round(score * 100, 2),
            "reference_number": reference_number,
            "uploaded_number": uploaded_number,
            "number_match": number_match,
            "authenticity": authenticity
        }
    else:  # cheque
        # Extract cheque text
        cheque_text = extract_text(file_path, 'cheque')
        authenticity = "Real Cheque" if score >= 0.80 else "Fake Cheque"
        
        result = {
            "ssim_score": round(score * 100, 2),
            "authenticity": authenticity,
            "cheque_text": cheque_text
        }

    # Save processed images
    output_paths = {
        "processed": os.path.join(UPLOAD_FOLDER[doc_type], f"processed_{file.filename}"),
        "difference": os.path.join(UPLOAD_FOLDER[doc_type], f"difference_{file.filename}"),
        "threshold": os.path.join(UPLOAD_FOLDER[doc_type], f"threshold_{file.filename}")
    }

    cv2.imwrite(output_paths["processed"], uploaded_img)
    cv2.imwrite(output_paths["difference"], diff)
    cv2.imwrite(output_paths["threshold"], thresh)

    # Add image paths to result
    result.update({
        "processed_image": f"/uploads/{doc_type}/{os.path.basename(output_paths['processed'])}",
        "difference_image": f"/uploads/{doc_type}/{os.path.basename(output_paths['difference'])}",
        "threshold_image": f"/uploads/{doc_type}/{os.path.basename(output_paths['threshold'])}"
    })

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=8080) 