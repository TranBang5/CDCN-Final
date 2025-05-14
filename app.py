from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models.database import db, User, Course, Tutor, Material, StudyPlan, SelectedCourse, SelectedTutor, SelectedMaterial
from models.recommender import RecommendationModel, load_and_preprocess_data
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import time
from sqlalchemy.exc import OperationalError
import re
import json

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'mysql+mysqlconnector://user:password@db:3306/recommendation_db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Model weights and BruteForce data paths
WEIGHTS_DIR = os.getenv('WEIGHTS_DIR', './checkpoints')
BRUTEFORCE_DATA_PATH = os.getenv('BRUTEFORCE_DATA_PATH', './bruteforce_data.npz')

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load preprocessed data and model
print("Loading preprocessed data...")
data = load_and_preprocess_data()
print("Data loaded successfully")

print("Initializing recommendation model...")
model = RecommendationModel(
    student_features=data['student_data'],
    course_features=data['course_data'],
    tutor_features=data['tutor_data'],
    material_features=data['material_data'],
    student_course_train=data['student_course_train'],
    student_tutor_train=data['student_tutor_train'],
    student_material_train=data['student_material_train'],
    subject_vocab=data['subject_vocab'],
    grade_vocab=data['grade_vocab'],
    material_type_vocab=data['material_type_vocab'],
    bruteforce_data_path=BRUTEFORCE_DATA_PATH
)
print("Model initialized successfully")

# Load model weights from checkpoint
checkpoint = tf.train.Checkpoint(model=model)
latest_checkpoint = tf.train.latest_checkpoint(WEIGHTS_DIR)
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    print(f"Loaded pre-trained model weights from {latest_checkpoint}")
else:
    print("Warning: No checkpoint found in", WEIGHTS_DIR)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        password = request.form['password']
        school = request.form.get('school')
        current_grade = request.form.get('current_grade')
        learning_goals = request.form.get('learning_goals')
        favorite_subjects = request.form.get('favorite_subjects')
        preferred_learning_method = request.form.get('preferred_learning_method')
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(
            full_name=full_name,
            email=email,
            password_hash=hashed_password,
            school=school,
            current_grade=current_grade,
            learning_goals=learning_goals,
            favorite_subjects=favorite_subjects,
            preferred_learning_method=preferred_learning_method
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('profile'))
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.school = request.form.get('school')
        current_user.current_grade = request.form.get('current_grade')
        current_user.favorite_subjects = request.form.get('favorite_subjects')
        current_user.learning_goals = request.form.get('learning_goals')
        current_user.preferred_learning_method = request.form.get('preferred_learning_method')
        db.session.commit()
        flash('Profile updated successfully')
        return redirect(url_for('profile'))
    return render_template('profile.html', user=current_user)

@app.route('/recommendations', methods=['GET', 'POST'])
@login_required
def recommendations():
    if not all([current_user.school, current_user.current_grade, current_user.favorite_subjects, current_user.learning_goals]):
        flash('Please complete your profile before viewing recommendations')
        return redirect(url_for('profile'))
    
    # Prepare user features for the model
    user_features = {
        'Trường học hiện tại': str(current_user.school).strip().lower(),
        'Khối Lớp hiện tại': str(current_user.current_grade).strip().lower(),
        'Mục tiêu học': str(current_user.learning_goals).strip().lower(),
        'Môn học yêu thích': str(current_user.favorite_subjects).strip().lower(),
        'Phương pháp học yêu thích': str(current_user.preferred_learning_method).strip().lower()
    }
    
    # Convert user features to model input format
    user_data = {k: tf.convert_to_tensor([v.encode('utf-8')], dtype=tf.string) for k, v in user_features.items()}
    user_dataset = tf.data.Dataset.from_tensor_slices(user_data).batch(1)
    
    try:
        # Get recommendations from model
        for batch in user_dataset:
            student_embeddings = model.student_model(batch)
            scores, top_k_ids = model.streaming(student_embeddings, k=10)
            top_k_ids = top_k_ids.numpy().astype(str)
            break
    except Exception as e:
        flash(f'Error generating recommendations: {str(e)}')
        return redirect(url_for('profile'))
    
    recommendations = {'courses': [], 'tutors': [], 'materials': []}
    
    # Apply filters if provided
    filters = {}
    if request.method == 'POST':
        filters = {
            'subject': request.form.get('subject'),
            'grade': request.form.get('grade'),
            'method': request.form.get('method')
        }
    
    # Process recommendations
    for id_ in top_k_ids[0]:
        if id_.startswith('course_'):
            course_id = int(id_.replace('course_', ''))
            course = Course.query.get(course_id)
            if course:
                # Apply filters
                if filters.get('subject') and course.subject != filters['subject']:
                    continue
                if filters.get('grade') and course.grade_level != filters['grade']:
                    continue
                if filters.get('method') and course.teaching_method != filters['method']:
                    continue
                recommendations['courses'].append(course)
        elif id_.startswith('tutor_'):
            tutor_id = int(id_.replace('tutor_', ''))
            tutor = Tutor.query.get(tutor_id)
            if tutor:
                # Apply filters
                if filters.get('subject') and tutor.subject != filters['subject']:
                    continue
                if filters.get('grade') and tutor.specialized_grade != filters['grade']:
                    continue
                if filters.get('method') and tutor.teaching_method != filters['method']:
                    continue
                recommendations['tutors'].append(tutor)
        elif id_.startswith('material_'):
            material_id = int(id_.replace('material_', ''))
            material = Material.query.get(material_id)
            if material:
                # Apply filters
                if filters.get('subject') and material.subject != filters['subject']:
                    continue
                if filters.get('grade') and material.grade_level != filters['grade']:
                    continue
                recommendations['materials'].append(material)
    
    if request.method == 'POST':
        # For AJAX requests, return only the recommendations container
        return render_template('_recommendations.html', recommendations=recommendations)
    
    return render_template('recommendations.html', recommendations=recommendations)

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/study_plan', methods=['GET', 'POST'])
@login_required
def study_plan():
    # Get or create study plan for user
    study_plan = current_user.study_plan
    if not study_plan:
        study_plan = StudyPlan(user_id=current_user.id)
        db.session.add(study_plan)
        db.session.commit()

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'add_course':
            course_id = request.form.get('course_id')
            time_slot = request.form.get('time_slot')
            if not (course_id and time_slot):
                flash('Missing course or time slot')
                return redirect(url_for('study_plan'))

            # Validate time slot format (e.g., "7h45-10h10 thứ 3")
            if not re.match(r'^\d{1,2}h\d{0,2}-\d{1,2}h\d{0,2} thứ [2-7]$', time_slot):
                flash('Invalid time slot format for course')
                return redirect(url_for('study_plan'))

            # Check for time slot conflicts
            if check_time_conflict(study_plan, time_slot):
                flash('Time slot conflicts with existing schedule')
                return redirect(url_for('study_plan'))

            selected_course = SelectedCourse(
                study_plan_id=study_plan.id,
                course_id=course_id,
                time_slot=time_slot
            )
            db.session.add(selected_course)
            db.session.commit()
            flash('Course added to study plan')

        elif action == 'add_tutor':
            tutor_id = request.form.get('tutor_id')
            selected_time_slot = request.form.get('selected_time_slot')
            if not (tutor_id and selected_time_slot):
                flash('Missing tutor or time slot')
                return redirect(url_for('study_plan'))

            # Validate time slot format (e.g., "18h thứ 3")
            if not re.match(r'^\d{1,2}h\d{0,2} thứ [2-7]$', selected_time_slot):
                flash('Invalid time slot format for tutor')
                return redirect(url_for('study_plan'))

            # Validate against tutor's available time slots
            tutor = Tutor.query.get(tutor_id)
            if not tutor or not is_valid_tutor_time_slot(tutor.teaching_time, selected_time_slot):
                flash('Selected time slot not available for this tutor')
                return redirect(url_for('study_plan'))

            # Check for time slot conflicts
            if check_time_conflict(study_plan, selected_time_slot):
                flash('Time slot conflicts with existing schedule')
                return redirect(url_for('study_plan'))

            selected_tutor = SelectedTutor(
                study_plan_id=study_plan.id,
                tutor_id=tutor_id,
                selected_time_slot=selected_time_slot
            )
            db.session.add(selected_tutor)
            db.session.commit()
            flash('Tutor added to study plan')

        elif action == 'add_material':
            material_id = request.form.get('material_id')
            time_slots = request.form.get('time_slots')
            if not (material_id and time_slots):
                flash('Missing material or time slots')
                return redirect(url_for('study_plan'))

            try:
                time_slots_list = json.loads(time_slots)
                # Validate each time slot (e.g., "7h-10h thứ 3")
                for slot in time_slots_list:
                    if not re.match(r'^\d{1,2}h\d{0,2}-\d{1,2}h\d{0,2} thứ [2-7]$', slot):
                        flash(f'Invalid time slot format: {slot}')
                        return redirect(url_for('study_plan'))

                    # Check for time slot conflicts
                    if check_time_conflict(study_plan, slot):
                        flash(f'Time slot conflicts with existing schedule: {slot}')
                        return redirect(url_for('study_plan'))

                selected_material = SelectedMaterial(
                    study_plan_id=study_plan.id,
                    material_id=material_id,
                    time_slots=time_slots
                )
                db.session.add(selected_material)
                db.session.commit()
                flash('Material added to study plan')
            except json.JSONDecodeError:
                flash('Invalid time slots format')
                return redirect(url_for('study_plan'))

        elif action == 'remove_item':
            item_type = request.form.get('item_type')
            item_id = request.form.get('item_id')
            if item_type and item_id:
                if item_type == 'course':
                    SelectedCourse.query.filter_by(id=item_id, study_plan_id=study_plan.id).delete()
                elif item_type == 'tutor':
                    SelectedTutor.query.filter_by(id=item_id, study_plan_id=study_plan.id).delete()
                elif item_type == 'material':
                    SelectedMaterial.query.filter_by(id=item_id, study_plan_id=study_plan.id).delete()
                db.session.commit()
                flash('Item removed from study plan')

        return redirect(url_for('study_plan'))

    # Get all selected items
    selected_courses = study_plan.selected_courses
    selected_tutors = study_plan.selected_tutors
    selected_materials = study_plan.selected_materials

    # Get all available items for selection
    courses = Course.query.all()
    tutors = Tutor.query.all()
    materials = Material.query.all()

    # Sort items by time
    sorted_items = sort_items_by_time(selected_courses, selected_tutors, selected_materials)

    return render_template('study_plan.html',
                         study_plan=study_plan,
                         selected_courses=selected_courses,
                         selected_tutors=selected_tutors,
                         selected_materials=selected_materials,
                         courses=courses,
                         tutors=tutors,
                         materials=materials,
                         sorted_items=sorted_items)

def parse_time_slot(time_slot):
    """Parse time slot string into start time (minutes), end time (minutes), and day (integer)."""
    try:
        day_map = {
            'thứ 2': 2, 'thứ hai': 2, '2': 2,
            'thứ 3': 3, 'thứ ba': 3, '3': 3,
            'thứ 4': 4, 'thứ tư': 4, '4': 4,
            'thứ 5': 5, 'thứ năm': 5, '5': 5,
            'thứ 6': 6, 'thứ sáu': 6, '6': 6,
            'thứ 7': 7, 'thứ bảy': 7, '7': 7
        }

        # Split time and day
        time_part, day_part = time_slot.lower().split(' thứ ')
        day = day_map.get(day_part.strip())
        if not day:
            return None, None, None

        # Parse time
        if '-' in time_part:
            start, end = time_part.split('-')
            start_minutes = time_to_minutes(start)
            end_minutes = time_to_minutes(end)
        else:
            start_minutes = time_to_minutes(time_part)
            end_minutes = start_minutes + 60  # Assume 1-hour duration for tutors
        return start_minutes, end_minutes, day
    except (ValueError, KeyError):
        return None, None, None

def time_to_minutes(time_str):
    """Convert time string (e.g., '7h45') to minutes since midnight."""
    time_str = time_str.replace('h', ':')
    hours, minutes = map(int, time_str.split(':') if ':' in time_str else [time48, '0'])
    return hours * 60 + minutes

def is_valid_tutor_time_slot(available_slots, selected_slot):
    """Check if selected time slot is available for the tutor."""
    available_slots = available_slots.split(';')
    selected_start, _, selected_day = parse_time_slot(selected_slot)
    for slot in available_slots:
        slot_start, _, slot_day = parse_time_slot(slot)
        if selected_start == slot_start and selected_day == slot_day:
            return True
    return False

def check_time_conflict(study_plan, new_time_slot):
    """Check if a new time slot conflicts with existing schedule."""
    new_start, new_end, new_day = parse_time_slot(new_time_slot)
    if new_start is None:
        return True  # Invalid time slot

    # Organize existing slots by day
    day_slots = {}
    for course in study_plan.selected_courses:
        start, end, day = parse_time_slot(course.time_slot)
        if start is not None:
            day_slots.setdefault(day, []).append((start, end))

    for tutor in study_plan.selected_tutors:
        start, end, day = parse_time_slot(tutor.selected_time_slot)
        if start is not None:
            day_slots.setdefault(day, []).append((start, end))

    for material in study_plan.selected_materials:
        time_slots = json.loads(material.time_slots)
        for slot in time_slots:
            start, end, day = parse_time_slot(slot)
            if start is not None:
                day_slots.setdefault(day, []).append((start, end))

    # Check for conflicts on the same day
    if new_day in day_slots:
        for start, end in day_slots[new_day]:
            if new_start < end and new_end > start:
                return True
    return False

def sort_items_by_time(selected_courses, selected_tutors, selected_materials):
    """Sort all selected items by day and precise start time."""
    items = []

    # Add courses
    for course in selected_courses:
        start, end, day = parse_time_slot(course.time_slot)
        if start is not None:
            items.append({
                'type': 'course',
                'item': course,
                'start': start,
                'end': end,
                'day': day
            })

    # Add tutors
    for tutor in selected_tutors:
        start, end, day = parse_time_slot(tutor.selected_time_slot)
        if start is not None:
            items.append({
                'type': 'tutor',
                'item': tutor,
                'start': start,
                'end': end,
                'day': day
            })

    # Add materials
    for material in selected_materials:
        time_slots = json.loads(material.time_slots)
        for slot in time_slots:
            start, end, day = parse_time_slot(slot)
            if start is not None:
                items.append({
                    'type': 'material',
                    'item': material,
                    'start': start,
                    'end': end,
                    'day': day
                })

    # Sort by day, start time, and type (course > tutor > material)
    return sorted(items, key=lambda x: (x['day'], x['start'], {'course': 0, 'tutor': 1, 'material': 2}[x['type']]))

def init_db():
    max_retries = 5
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            db.create_all()
            print("Database initialized successfully")
            return
        except OperationalError as e:
            if attempt < max_retries - 1:
                print(f"Database connection failed. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Failed to connect to database after multiple attempts")
                raise e

if __name__ == '__main__':
    with app.app_context():
        init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)