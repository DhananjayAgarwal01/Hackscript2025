# Document Forgery Detection System

A modern web application for verifying document authenticity using advanced AI and machine learning techniques.

## Features

- Document upload and verification
- Multiple document format support (PDF, PNG, JPG)
- Advanced forgery detection
- Real-time analysis
- Detailed verification reports
- History tracking

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize the database:
```bash
python init_db.py
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Requirements

- Python 3.8 or higher
- Tesseract OCR
- OpenCV
- SQLite

## Technology Stack

- Flask (Web Framework)
- SQLAlchemy (Database ORM)
- OpenCV (Image Processing)
- Tesseract (OCR)
- Bootstrap 5 (UI Framework)

## License

MIT License
