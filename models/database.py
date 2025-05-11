from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Student profile
    school = db.Column(db.String(100))
    current_grade = db.Column(db.String(20))
    favorite_subjects = db.Column(db.String(200))
    learning_goals = db.Column(db.Text)
    
    # Relationships
    study_plans = db.relationship('StudyPlan', backref='student', lazy=True)

class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    center_name = db.Column(db.String(100), nullable=False)
    subject = db.Column(db.String(50), nullable=False)
    grade_level = db.Column(db.String(20), nullable=False)
    schedule = db.Column(db.String(100))
    address = db.Column(db.String(200))
    teaching_method = db.Column(db.String(20))  # online/offline
    cost = db.Column(db.Float)
    
    # Relationships
    study_plans = db.relationship('StudyPlanCourse', backref='course', lazy=True)

class Tutor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    subject = db.Column(db.String(50), nullable=False)
    schedule = db.Column(db.String(100))
    specialized_grade = db.Column(db.String(20))
    teaching_experience = db.Column(db.Integer)  # years
    teaching_method = db.Column(db.String(20))  # online/offline
    
    # Relationships
    study_plans = db.relationship('StudyPlanTutor', backref='tutor', lazy=True)

class Material(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    subject = db.Column(db.String(50), nullable=False)
    grade_level = db.Column(db.String(20))
    material_type = db.Column(db.String(20))  # paper/digital
    
    # Relationships
    study_plans = db.relationship('StudyPlanMaterial', backref='material', lazy=True)

class StudyPlan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    description = db.Column(db.Text)
    
    # Relationships
    courses = db.relationship('StudyPlanCourse', backref='study_plan', lazy=True)
    tutors = db.relationship('StudyPlanTutor', backref='study_plan', lazy=True)
    materials = db.relationship('StudyPlanMaterial', backref='study_plan', lazy=True)

class StudyPlanCourse(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    study_plan_id = db.Column(db.Integer, db.ForeignKey('study_plan.id'), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)

class StudyPlanTutor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    study_plan_id = db.Column(db.Integer, db.ForeignKey('study_plan.id'), nullable=False)
    tutor_id = db.Column(db.Integer, db.ForeignKey('tutor.id'), nullable=False)

class StudyPlanMaterial(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    study_plan_id = db.Column(db.Integer, db.ForeignKey('study_plan.id'), nullable=False)
    material_id = db.Column(db.Integer, db.ForeignKey('material.id'), nullable=False) 