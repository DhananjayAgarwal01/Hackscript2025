from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from ideathon.models.record import Record

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)

@app.route('/api/delete-all-records', methods=['DELETE'])
def delete_all_records():
    try:
        db.session.query(Record).delete()
        db.session.commit()
        return jsonify({'message': 'All records deleted successfully'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)