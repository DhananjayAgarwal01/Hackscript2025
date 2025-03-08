# ...existing code...

class Record(db.Model):
    __tablename__ = 'records'
    id = db.Column(db.Integer, primary_key=True)
    # ...existing code...
    
    @classmethod
    def delete_all(cls):
        cls.query.delete()
        db.session.commit()

# ...existing code...
