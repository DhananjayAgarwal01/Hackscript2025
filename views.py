from flask import Flask, request, redirect, flash, jsonify
from werkzeug.utils import secure_filename
from .core.verification import DocumentVerifier
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
REFERENCE_PAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'reference', 'reference_pan.png')

verifier = DocumentVerifier(UPLOAD_FOLDER)

@app.route('/verify-document', methods=['POST'])
def verify_document():
    if 'document' not in request.files:
        flash('No file uploaded')
        return redirect(request.url)
    
    file = request.files['document']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        result = verifier.verify_document(REFERENCE_PAN, filepath)
        
        return jsonify({
            'status': 'success',
            'data': {
                'score': result['ssim_score'],
                'is_authentic': result['is_authentic'],
                'result_image': result['result_image'].replace(app.root_path, '').replace('\\', '/')
            }
        })

if __name__ == '__main__':
    app.run(debug=True)
