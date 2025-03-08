# ...existing code...

from .utils.pan_validator import PANValidator
from flask import current_app, jsonify, flash
import logging
import os
import shutil

# ...existing code...

@app.route('/process_document', methods=['POST'])
def process_document():
    if 'document' not in request.files:
        return jsonify({'error': 'No document provided'}), 400
    
    file = request.files['document']
    document_type = request.form.get('document_type')
    
    if document_type == 'pan_card':
        # Save the file temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(temp_path)
        
        # Verify PAN card
        is_valid, message, confidence = PANValidator.verify_pan_card(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        result = {
            'validation_result': is_valid,
            'message': message,
            'confidence_score': confidence
        }
        
        # Store analysis results
        document = Document(
            filename=secure_filename(file.filename),
            document_type=document_type,
            hash_value=compute_file_hash(file),
            forgery_score=1.0 - confidence
        )
        db.session.add(document)
        
        analysis_result = AnalysisResult(
            document=document,
            analysis_type='pan_verification',
            result=json.dumps(result),
            confidence_score=confidence
        )
        db.session.add(analysis_result)
        db.session.commit()
        
        return jsonify(result)
    
    # ...existing code for other document types...

# ...existing code...

@app.route('/delete_document/<filename>', methods=['DELETE'])
def delete_document(filename):
    try:
        upload_folder = current_app.config['UPLOAD_FOLDER']
        secure_name = secure_filename(filename)
        file_path = os.path.join(upload_folder, secure_name)
        json_path = os.path.join(upload_folder, f"{secure_name}.json")
        
        deleted = False
        
        # Delete document file
        if os.path.exists(file_path):
            os.remove(file_path)
            deleted = True
            
        # Delete associated JSON file
        if os.path.exists(json_path):
            os.remove(json_path)
            deleted = True
        
        if deleted:
            current_app.logger.info(f"Successfully deleted document: {filename}")
            return jsonify({
                'success': True,
                'message': 'Document deleted successfully'
            })
        
        return jsonify({
            'success': False,
            'message': 'Document not found'
        }), 404
        
    except Exception as e:
        current_app.logger.error(f"Error deleting document {filename}: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Failed to delete document: {str(e)}'
        }), 500

@app.route('/delete_all_documents', methods=['DELETE'])
def delete_all_documents():
    try:
        upload_folder = current_app.config['UPLOAD_FOLDER']
        
        # Create a new empty uploads folder
        if os.path.exists(upload_folder):
            shutil.rmtree(upload_folder)
        os.makedirs(upload_folder)
        
        current_app.logger.info("Successfully deleted all documents")
        return jsonify({
            'success': True,
            'message': 'All documents deleted successfully'
        })
        
    except Exception as e:
        current_app.logger.error(f"Error deleting all documents: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Failed to delete all documents: {str(e)}'
        }), 500

# ...existing code...
