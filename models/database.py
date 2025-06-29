from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Student profile
    school = db.Column(db.String(200))
    current_grade = db.Column(db.String(50))
    favorite_subjects = db.Column(db.String(500))
    learning_goals = db.Column(db.String(200))
    preferred_learning_method = db.Column(db.String(200))
    
    # Relationships
    study_plan = db.relationship('StudyPlan', backref='user', uselist=False, cascade='all, delete-orphan')
    feedback = db.relationship('Feedback', backref='user', lazy=True, cascade='all, delete-orphan')

    def get_id(self):
        return str(self.id)

    def __repr__(self):
        return f'<User {self.email}>'

class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    subject = db.Column(db.String(50), nullable=False)
    grade_level = db.Column(db.String(20), nullable=False)
    schedule = db.Column(db.String(100))
    address = db.Column(db.String(200))
    teaching_method = db.Column(db.String(20))  # online/offline
    cost = db.Column(db.Float)
    teaching_time = db.Column(db.String(100))  # Thêm trường để lưu thời gian giảng dạy  
    location = db.Column(db.String(200))  # Thêm trường để lưu địa điểm
    
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
    teaching_time = db.Column(db.String(200))  # Thêm trường để lưu thời gian giảng dạy
    experience = db.Column(db.Integer)  # Thêm trường để lưu kinh nghiệm
    
    # Relationships
    study_plans = db.relationship('StudyPlanTutor', backref='tutor', lazy=True)

class Material(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    subject = db.Column(db.String(50), nullable=False)
    grade_level = db.Column(db.String(20))
    material_type = db.Column(db.String(20))  # paper/digital
    description = db.Column(db.Text)  # Thêm trường để lưu mô tả
    
    # Relationships
    study_plans = db.relationship('StudyPlanMaterial', backref='material', lazy=True)

class StudyPlan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    selected_courses = db.relationship('SelectedCourse', backref='study_plan', lazy=True, cascade='all, delete-orphan')
    selected_tutors = db.relationship('SelectedTutor', backref='study_plan', lazy=True, cascade='all, delete-orphan')
    selected_materials = db.relationship('SelectedMaterial', backref='study_plan', lazy=True, cascade='all, delete-orphan')

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

class SelectedCourse(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    study_plan_id = db.Column(db.Integer, db.ForeignKey('study_plan.id'), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    time_slot = db.Column(db.String(50), nullable=False)  # Format: "7h45-10h10 thứ 3"
    
    # Relationship
    course = db.relationship('Course', backref='selected_courses')

class SelectedTutor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    study_plan_id = db.Column(db.Integer, db.ForeignKey('study_plan.id'), nullable=False)
    tutor_id = db.Column(db.Integer, db.ForeignKey('tutor.id'), nullable=False)
    selected_time_slot = db.Column(db.String(50), nullable=False)  # Format: "18h thứ ba"
    
    # Relationship
    tutor = db.relationship('Tutor', backref='selected_tutors')

class SelectedMaterial(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    study_plan_id = db.Column(db.Integer, db.ForeignKey('study_plan.id'), nullable=False)
    material_id = db.Column(db.Integer, db.ForeignKey('material.id'), nullable=False)
    time_slots = db.Column(db.Text, nullable=False)  # JSON string of time slots
    
    # Relationship
    material = db.relationship('Material', backref='selected_materials')

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    feedback_type = db.Column(db.String(50), nullable=False)  # 'general', 'recommendation', 'interface', etc.
    content = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Integer)  # 1-5 star rating
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Feedback {self.id} from User {self.user_id}>' 