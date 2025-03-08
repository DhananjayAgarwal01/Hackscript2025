from app import app, db

with app.app_context():
    # Create all database tables
    db.create_all()
    print("Database tables created successfully!") 