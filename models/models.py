from datetime import datetime
from models.database import db

class StudyPlanItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    item_type = db.Column(db.String(20), nullable=False)  # 'course', 'tutor', or 'material'
    item_id = db.Column(db.Integer, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    subject = db.Column(db.String(50), nullable=False)
    grade = db.Column(db.String(20), nullable=False)
    method = db.Column(db.String(20))  # Optional, for courses and tutors
    time_slots = db.Column(db.String(500))  # JSON string of time slots
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('study_plan_items', lazy=True)) 