from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models.database import db, User, Course, Tutor, Material, StudyPlan
from models.recommender import RecommendationModel, load_and_preprocess_data
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'mysql://user:password@mysql:3306/recommendation_db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Data and model paths
DATA_DIR = os.getenv('DATA_DIR', './data')
WEIGHTS_PATH = os.getenv('WEIGHTS_PATH', './recommendation_model_weights.weights.h5')

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load preprocessed data and model
data = load_and_preprocess_data(data_dir=DATA_DIR)
model = RecommendationModel(
    student_features=data['student_data'],
    course_features=data['course_data'],
    tutor_features=data['tutor_data'],
    material_features=data['material_data'],
    student_course_train=data['student_course_train'],
    student_tutor_train=data['student_tutor_train'],
    student_material_train=data['student_material_train']
)
model.load_weights(WEIGHTS_PATH)
print("Loaded pre-trained model weights")

# Sync database with CSV data if empty
def sync_data():
    course_data = pd.read_csv(os.path.join(DATA_DIR, 'trung_tam.csv'))
    tutor_data = pd.read_csv(os.path.join(DATA_DIR, 'gia_su.csv'))
    material_data = pd.read_csv(os.path.join(DATA_DIR, 'tai_lieu.csv'))

    with app.app_context():
        if not Course.query.first():
            for _, row in course_data.iterrows():
                course = Course(
                    id=int(row['ID Trung Tâm']),
                    name=row['Tên Trung Tâm'],
                    subject=row['Môn học'],
                    grade=row['Khối Lớp'],
                    method=row['Phương pháp học'],
                    cost=float(row['Chi phí']) if pd.notna(row['Chi phí']) else 0.0,
                    duration=float(row['Thời gian']) if pd.notna(row['Thời gian']) else 0.0,
                    address=row['Địa chỉ'],
                    rating=float(row['Đánh giá']) if pd.notna(row['Đánh giá']) else 0.0
                )
                db.session.merge(course)

            for _, row in tutor_data.iterrows():
                tutor = Tutor(
                    id=int(row['ID Gia Sư']),
                    name=row['Tên gia sư'],
                    subject=row['Môn học'],
                    grade=row['Khối Lớp'],
                    teaching_time=row['Thời gian dạy học'],
                    experience=float(row['Kinh nghiệm giảng dạy']) if pd.notna(row['Kinh nghiệm giảng dạy']) else 0.0
                )
                db.session.merge(tutor)

            for _, row in material_data.iterrows():
                material = Material(
                    id=int(row['ID Tài Liệu']),
                    name=row['Tên tài liệu'],
                    subject=row['Môn học'],
                    grade=row['Khối Lớp'],
                    type=row['Loại tài liệu']
                )
                db.session.merge(material)

            db.session.commit()
            print("Data synced to database")

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
        current_user.preferred_learning_method = request.form.get('preferred_learning_method', '')
        
        db.session.commit()
        flash('Profile updated successfully')
        return redirect(url_for('dashboard'))
    
    return render_template('profile.html')

@app.route('/recommendations')
@login_required
def recommendations():
    # Check if user profile is complete
    if not all([current_user.school, current_user.current_grade, current_user.favorite_subjects, current_user.learning_goals]):
        flash('Please complete your profile before viewing recommendations')
        return redirect(url_for('profile'))
    
    # Prepare user features
    user_features = {
        'Trường học hiện tại': 'cat_' + str(current_user.school).strip().lower(),
        'Khối Lớp hiện tại': 'cat_' + str(current_user.current_grade).strip().lower(),
        'Mục tiêu học': 'cat_' + str(current_user.learning_goals).strip().lower(),
        'Môn học yêu thích': 'cat_' + str(current_user.favorite_subjects).strip().lower(),
        'Phương pháp học yêu thích': 'cat_' + str(current_user.preferred_learning_method).strip().lower()
    }
    
    # Convert to TensorFlow dataset
    user_data = {k: tf.convert_to_tensor([v.encode('utf-8')], dtype=tf.string) for k, v in user_features.items()}
    user_dataset = tf.data.Dataset.from_tensor_slices(user_data).batch(1)
    
    # Generate recommendations
    try:
        for batch in user_dataset:
            student_embeddings = model.student_model(batch)
            scores, top_k_ids = model.streaming(student_embeddings, k=10)
            top_k_ids = top_k_ids.numpy().astype(str)
            break
        
        # Parse recommendations
        recommendations = {'courses': [], 'tutors': [], 'materials': []}
        for id_ in top_k_ids[0]:
            if id_.startswith('course_'):
                course_id = int(id_.replace('course_', ''))
                course = Course.query.get(course_id)
                if course:
                    recommendations['courses'].append({
                        'id': course.id,
                        'name': course.name,
                        'subject': course.subject,
                        'grade': course.grade
                    })
            elif id_.startswith('tutor_'):
                tutor_id = int(id_.replace('tutor_', ''))
                tutor = Tutor.query.get(tutor_id)
                if tutor:
                    recommendations['tutors'].append({
                        'id': tutor.id,
                        'name': tutor.name,
                        'subject': tutor.subject,
                        'grade': tutor.grade
                    })
            elif id_.startswith('material_'):
                material_id = int(id_.replace('material_', ''))
                material = Material.query.get(material_id)
                if material:
                    recommendations['materials'].append({
                        'id': material.id,
                        'name': material.name,
                        'subject': material.subject,
                        'type': material.type
                    })
        
        return render_template('recommendations.html', recommendations=recommendations)
    
    except Exception as e:
        flash(f'Error generating recommendations: {str(e)}')
        return redirect(url_for('dashboard'))

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
            course = Course.query.get(int(course_id))
            if course:
                study_plan.courses.append(course)
        
        for tutor_id in tutor_ids:
            tutor = Tutor.query.get(int(tutor_id))
            if tutor:
                study_plan.tutors.append(tutor)
        
        for material_id in material_ids:
            material = Course.query.get(int(material_id))
            if material:
                study_plan.materials.append(material)
        
        db.session.commit()
        flash('Study plan created successfully')
        return redirect(url_for('dashboard'))
    
    # Fetch available items for selection
    courses = Course.query.all()
    tutors = Tutor.query.all()
    materials = Material.query.all()
    return render_template('study_plan.html', courses=courses, tutors=tutors, materials=materials)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        sync_data()
    app.run(host='0.0.0.0', port=5000, debug=True)
