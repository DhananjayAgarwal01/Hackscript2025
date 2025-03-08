import os
import sys

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from ideathon.models.record import Record

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
    db = SQLAlchemy(app)
    return app, db

def delete_all_records_from_db():
    app, db = create_app()
    with app.app_context():
        try:
            # Delete all records
            Record.query.delete()
            db.session.commit()
            print("All records deleted successfully from database.")
        except Exception as e:
            print(f"Error deleting records: {str(e)}")
            db.session.rollback()

def delete_database_file():
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test.db')
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
            print("Database file deleted successfully.")
        else:
            print("Database file doesn't exist.")
    except Exception as e:
        print(f"Error deleting database file: {str(e)}")

if __name__ == '__main__':
    choice = input("Enter 1 to delete all records, 2 to delete database file, 3 for both: ")
    if choice in ('1', '3'):
        delete_all_records_from_db()
    if choice in ('2', '3'):
        delete_database_file()
