# Document Verification System

A unified system for verifying multiple types of documents including Aadhaar cards, passports, cheques, and bank statements using image processing and similarity analysis.

## Features

- Supports multiple document types:
  - Aadhaar Card Verification
  - Passport Verification
  - Cheque Verification
  - Bank Statement Verification
- Real-time document authenticity checking
- Visual difference highlighting
- Confidence score calculation
- OCR text extraction (for Aadhaar cards and cheques)
- Face detection (for Aadhaar cards)
- Modern web interface with real-time results

## Prerequisites

1. Python 3.7 or higher
2. Tesseract OCR engine
3. Required reference images:
   - `reference_aadhaar.png` - Reference Aadhaar card image
   - `1.jpg` - Reference passport image
   - `cheque_og.png` - Reference cheque image
   - Bank statement references:
     - Real samples: `1.jpg`, `3.jpg`
     - Fake samples: `Fake.png`, `f2.webp`

## Installation

1. Install Tesseract OCR:
   - Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - Mac: `brew install tesseract`

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place all reference images in the same directory as the Python files.

5. Update Tesseract path in `unified_app.py` if necessary:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path
   ```

## Usage

1. Start the Flask application:
   ```bash
   python unified_app.py
   ```

2. Open a web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Select the document type and upload the document you want to verify.

4. View the results:
   - Authenticity status
   - Confidence score
   - Processed images
   - Difference maps
   - Additional information based on document type

## Confidence Thresholds

- Aadhaar Card: 80% similarity required for "Real" status
- Passport: 90% similarity required for "Real" status
- Cheque: 80% similarity required for "Real" status
- Bank Statement: 80% similarity required for "Real" status

## Directory Structure

```
.
├── unified_app.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
├── uploads/
└── reference_images/
    ├── reference_aadhaar.png
    ├── 1.jpg
    ├── 3.jpg
    ├── cheque_og.png
    ├── Fake.png
    └── f2.webp
```

## Security Notes

1. The system stores uploaded files temporarily in the `uploads` directory.
2. Processed images are stored in the `static` directory.
3. Both directories are created automatically if they don't exist.
4. It's recommended to implement additional security measures for production use.

## Contributing

Feel free to submit issues and enhancement requests!
