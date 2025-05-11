from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models.database import db, User, Course, Tutor, Material, StudyPlan
from models.recommender import RecommendationModel
import tensorflow as tf
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'mysql://user:password@localhost/recommendation_db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
        
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.school = request.form['school']
        current_user.current_grade = request.form['current_grade']
        current_user.favorite_subjects = request.form['favorite_subjects']
        current_user.learning_goals = request.form['learning_goals']
        
        db.session.commit()
        flash('Profile updated successfully')
        return redirect(url_for('dashboard'))
    
    return render_template('profile.html')

@app.route('/recommendations')
@login_required
def recommendations():
    # Load and preprocess data
    student_data = pd.read_csv('data/hoc_sinh.csv')
    course_data = pd.read_csv('data/trung_tam.csv')
    tutor_data = pd.read_csv('data/gia_su.csv')
    material_data = pd.read_csv('data/tai_lieu.csv')
    
    # Initialize and train the model
    model = RecommendationModel(
        student_features=student_data,
        course_features=course_data,
        tutor_features=tutor_data,
        material_features=material_data
    )
    
    # Get recommendations for current user
    user_features = {
        'school': current_user.school,
        'grade': current_user.current_grade,
        'favorite_subject': current_user.favorite_subjects
    }
    
    recommendations = model.predict(user_features)
    
    return render_template('recommendations.html', recommendations=recommendations)

@app.route('/study-plan', methods=['GET', 'POST'])
@login_required
def study_plan():
    if request.method == 'POST':
        description = request.form['description']
        course_ids = request.form.getlist('courses')
        tutor_ids = request.form.getlist('tutors')
        material_ids = request.form.getlist('materials')
        
        study_plan = StudyPlan(
            student_id=current_user.id,
            description=description
        )
        db.session.add(study_plan)
        db.session.commit()
        
        # Add selected items to study plan
        for course_id in course_ids:
            study_plan.courses.append(Course.query.get(course_id))
        
        for tutor_id in tutor_ids:
            study_plan.tutors.append(Tutor.query.get(tutor_id))
        
        for material_id in material_ids:
            study_plan.materials.append(Material.query.get(material_id))
        
        db.session.commit()
        flash('Study plan created successfully')
        return redirect(url_for('dashboard'))
    
    return render_template('study_plan.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True) 