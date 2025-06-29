from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models.database import db, User, Course, Tutor, Material, StudyPlan, SelectedCourse, SelectedTutor, SelectedMaterial, Feedback
from models.models import StudyPlanItem
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
print("Đang tải dữ liệu đã xử lý trước...")
data = load_and_preprocess_data()
print("Dữ liệu đã tải thành công")

print("Khởi tạo mô hình đề xuất...")
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
    teaching_time_vocab=data['teaching_time_vocab'],
    bruteforce_data_path=BRUTEFORCE_DATA_PATH
)
print("Khởi tạo mô hình thành công")

# Load model weights from checkpoint
checkpoint = tf.train.Checkpoint(model=model)
latest_checkpoint = tf.train.latest_checkpoint(WEIGHTS_DIR)
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    print(f"Đã tải trọng số mô hình đã huấn luyện từ {latest_checkpoint}")
else:
    print("Cảnh báo: Không tìm thấy checkpoint nào trong", WEIGHTS_DIR)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
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
        flash('Vui lòng hoàn thành hồ sơ của bạn trước khi xem đề xuất')
        return redirect(url_for('profile'))
    
    # Prepare user features for the model
    user_features = {
        'truong_hoc_hien_tai': str(current_user.school).strip().lower(),
        'khoi_lop_hien_tai': str(current_user.current_grade).strip().lower(),
        'muc_tieu_hoc': str(current_user.learning_goals).strip().lower(),
        'mon_hoc_yeu_thich': str(current_user.favorite_subjects).strip().lower(),
        'phuong_phap_hoc_yeu_thich': str(current_user.preferred_learning_method).strip().lower()
    }
    
    # Convert user features to model input format
    user_data = {k: tf.convert_to_tensor([v.encode('utf-8')], dtype=tf.string) for k, v in user_features.items()}
    user_dataset = tf.data.Dataset.from_tensor_slices(user_data).batch(1)
    
    try:
        # Get recommendations from model
        for batch in user_dataset:
            student_embeddings = model.student_model(batch)
            scores, top_k_ids = model.bruteforce(student_embeddings)
            scores = scores.numpy()[0]  # Get scores for all items
            top_k_ids = top_k_ids.numpy().astype(str)[0]  # Get IDs for all items
            break
    except Exception as e:
        flash(f'Lỗi khi tạo đề xuất: {str(e)}')
        return redirect(url_for('profile'))
    
    # Get active tab from request
    active_tab = request.args.get('tab', 'courses')

    # Get filter parameters
    filters = {
        'subject': request.args.get('subject', ''),
        'grade': request.args.get('grade', ''),
        'method': request.args.get('method', '')
    }
    
    # Thêm tùy chọn lọc theo khối lớp phù hợp
    grade_filter_option = request.args.get('grade_filter', 'all')  # 'all' hoặc 'matching'
    
    # Get study plan items to check which items are already in the plan
    study_plan_items = StudyPlanItem.query.filter_by(user_id=current_user.id).all()
    items_in_plan = {
        'course': [int(item.item_id) for item in study_plan_items if item.item_type == 'course'],
        'tutor': [int(item.item_id) for item in study_plan_items if item.item_type == 'tutor'],
        'material': [int(item.item_id) for item in study_plan_items if item.item_type == 'material']
    }
    
    # Initialize recommendations dictionary
    recommendations = {'courses': [], 'tutors': [], 'materials': []}
    
    # Create a mapping of item IDs to their scores
    item_scores = {}
    for id_, score in zip(top_k_ids, scores):
        item_scores[id_] = float(score)
    
    # Get all items from database
    all_courses = Course.query.all()
    all_tutors = Tutor.query.all()
    all_materials = Material.query.all()
    
    # Get all unique subjects, grades, and methods for filter options
    all_subjects = set()
    all_grades = set()
    all_methods = set()
    
    # Hàm chuẩn hóa tên môn học
    def normalize_subject(subject):
        subject = subject.strip().lower()
        if "hoá" in subject:
            subject = subject.replace("hoá", "hóa")
        return subject.title()
    
    # Hàm tách các khối lớp và trả về list
    def split_grades(grade_level):
        if not grade_level:
            return []
        if "," in grade_level:
            return [g.strip() for g in grade_level.split(",")]
        return [grade_level.strip()]
    
    # Hàm chuẩn hóa định dạng khối lớp
    def normalize_grade(grade):
        if not grade:
            return ""
        grade = grade.strip().lower()
        # Trích xuất số lớp từ chuỗi
        if 'lớp' in grade:
            # Nếu chứa "lớp", chỉ lấy phần số
            grade_num = ''.join(filter(str.isdigit, grade))
            if grade_num:
                return grade_num
        # Nếu đã là số, trả về nguyên dạng
        if grade.isdigit():
            return grade
        return grade  # Trả về nguyên bản nếu không thể chuẩn hóa
    
    for course in all_courses:
        all_subjects.add(normalize_subject(course.subject))
        for grade in split_grades(course.grade_level):
            all_grades.add(grade)
        all_methods.add(course.teaching_method)
    
    for tutor in all_tutors:
        all_subjects.add(normalize_subject(tutor.subject))
        for grade in split_grades(tutor.specialized_grade):
            all_grades.add(grade)
        all_methods.add(tutor.teaching_method)
    
    for material in all_materials:
        all_subjects.add(normalize_subject(material.subject))
        for grade in split_grades(material.grade_level):
            all_grades.add(grade)
    
    # Process courses with grade and subject priority
    for course in all_courses:
        course_id = f'course_{course.id}'
        base_score = item_scores.get(course_id, 0.0)
        
        # Check if grade level matches user's current grade
        grade_match = False
        course_grades = course.grade_level.split(",") if "," in course.grade_level else [course.grade_level]
        course_grades = [g.strip() for g in course_grades]
        
        # Gán thuộc tính grade_match trước khi kiểm tra điều kiện
        course.grade_match = False  # Đặt mặc định là False
        
        # Kiểm tra chính xác nếu khối lớp người dùng có trong danh sách khối lớp của khóa học
        for grade in course_grades:
            if normalize_grade(grade) == normalize_grade(current_user.current_grade):
                course.grade_match = True
                grade_match = True
                break
                
        # Áp dụng trọng số sau khi kiểm tra khớp
        if grade_match:
            base_score *= 200.0  # Phương pháp 1: Tăng từ 50.0 lên 200.0
        else:
            base_score *= 0.0001  # Phương pháp 2: Giảm từ 0.005 xuống 0.0001
        
        # Bỏ qua các khoá học không cùng khối lớp nếu người dùng chọn chỉ hiển thị khối lớp phù hợp
        if grade_filter_option == 'matching' and not grade_match:
            continue
        
        # Check if subject matches user's favorite subjects
        subject_match = False
        course_subject = course.subject.lower().replace("hoá", "hóa").strip()
        user_subjects = [s.lower().replace("hoá", "hóa").strip() for s in current_user.favorite_subjects.split(',')]
        
        if course_subject in user_subjects:
            base_score *= 180.0
            subject_match = True
        else:
            base_score *= 0.00001
            
        if course.teaching_method == current_user.preferred_learning_method:
            base_score *= 1.2
        
        if apply_filters(course, filters):
            course.in_plan = course.id in items_in_plan['course']
            # Add debug info
            course.subject_match = subject_match
            recommendations['courses'].append((course, base_score))
    
    # Process tutors with grade and subject priority
    for tutor in all_tutors:
        tutor_id = f'tutor_{tutor.id}'
        base_score = item_scores.get(tutor_id, 0.0)
        
        # Check if grade level matches user's current grade
        grade_match = False
        tutor_grades = tutor.specialized_grade.split(",") if "," in tutor.specialized_grade else [tutor.specialized_grade]
        tutor_grades = [g.strip() for g in tutor_grades]
        
        # Gán thuộc tính grade_match trước khi kiểm tra điều kiện
        tutor.grade_match = False  # Đặt mặc định là False
        
        # Kiểm tra chính xác nếu khối lớp người dùng có trong danh sách khối lớp của gia sư
        for grade in tutor_grades:
            if normalize_grade(grade) == normalize_grade(current_user.current_grade):
                tutor.grade_match = True
                grade_match = True
                break
                
        # Áp dụng trọng số sau khi kiểm tra khớp
        if grade_match:
            base_score *= 200.0  # Phương pháp 1: Tăng từ 50.0 lên 200.0
        else:
            base_score *= 0.0001  # Phương pháp 2: Giảm từ 0.005 xuống 0.0001
            
        # Bỏ qua các gia sư không cùng khối lớp nếu người dùng chọn chỉ hiển thị khối lớp phù hợp
        if grade_filter_option == 'matching' and not grade_match:
            continue
            
        # Check if subject matches user's favorite subjects
        subject_match = False
        tutor_subject = tutor.subject.lower().replace("hoá", "hóa").strip()
        user_subjects = [s.lower().replace("hoá", "hóa").strip() for s in current_user.favorite_subjects.split(',')]
        
        if tutor_subject in user_subjects:
            base_score *= 180.0
            subject_match = True
        else:
            base_score *= 0.00001
            
        if tutor.teaching_method == current_user.preferred_learning_method:
            base_score *= 1.2
        
        if apply_filters(tutor, filters):
            tutor.in_plan = tutor.id in items_in_plan['tutor']
            # Add debug info
            tutor.subject_match = subject_match
            # Ensure experience is an integer - sửa đổi để chuyển đổi đúng thành số nguyên
            if hasattr(tutor, 'experience') and tutor.experience is not None:
                try:
                    # Convert to int to avoid any formatting issues - no string formatting
                    tutor.experience = int(tutor.experience)/1000
                except (ValueError, TypeError):
                    tutor.experience = 0  # Giá trị mặc định nếu không thể chuyển đổi
            recommendations['tutors'].append((tutor, base_score))
    
    # Process materials with grade and subject priority
    for material in all_materials:
        material_id = f'material_{material.id}'
        base_score = item_scores.get(material_id, 0.0)
        
        # Check if grade level matches user's current grade
        grade_match = False
        material_grades = material.grade_level.split(",") if "," in material.grade_level else [material.grade_level]
        material_grades = [g.strip() for g in material_grades]
        
        # Gán thuộc tính grade_match trước khi kiểm tra điều kiện
        material.grade_match = False  # Đặt mặc định là False
        
        # Kiểm tra chính xác nếu khối lớp người dùng có trong danh sách khối lớp của tài liệu
        for grade in material_grades:
            if normalize_grade(grade) == normalize_grade(current_user.current_grade):
                material.grade_match = True
                grade_match = True
                break
                
        # Áp dụng trọng số sau khi kiểm tra khớp
        if grade_match:
            base_score *= 200.0  # Phương pháp 1: Tăng từ 50.0 lên 200.0
        else:
            base_score *= 0.0001  # Phương pháp 2: Giảm từ 0.005 xuống 0.0001
        
        # Bỏ qua các tài liệu không cùng khối lớp nếu người dùng chọn chỉ hiển thị khối lớp phù hợp
        if grade_filter_option == 'matching' and not grade_match:
            continue
            
        # Check if subject matches user's favorite subjects
        subject_match = False
        material_subject = material.subject.lower().replace("hoá", "hóa").strip()
        user_subjects = [s.lower().replace("hoá", "hóa").strip() for s in current_user.favorite_subjects.split(',')]
        
        if material_subject in user_subjects:
            base_score *= 180.0
            subject_match = True
        else:
            base_score *= 0.00001
            
        if apply_filters(material, filters):
            # Add debug info
            material.in_plan = material.id in items_in_plan['material']
            # Add debug info
            material.subject_match = subject_match
            recommendations['materials'].append((material, base_score))
    
    # Thêm debug mới để kiểm tra xem các item có khớp grade hay không
    print(f"\nĐang kiểm tra đề xuất với khối lớp hiện tại: {current_user.current_grade} (Chuẩn hóa: {normalize_grade(current_user.current_grade)})")
    
    # In ví dụ về việc chuẩn hóa khối lớp
    print(f"Ví dụ chuẩn hóa khối lớp:")
    for example in ["Lớp 10", "10", "lớp 11", "Lớp12", "12"]:
        print(f"  {example} -> {normalize_grade(example)}")
    
    # Hiển thị một số khóa học và khối lớp tương ứng để kiểm tra
    print("\nKiểm tra khớp lớp:")
    for i, course in enumerate(all_courses[:3]):
        user_grade = normalize_grade(current_user.current_grade)
        course_grades = [normalize_grade(g) for g in split_grades(course.grade_level)]
        match = user_grade in course_grades
        print(f"  Khóa học {course.name}: user_grade={user_grade}, course_grades={course_grades}, khớp={match}")
    
    # Đảm bảo các item có grade_match=True luôn hiển thị đầu tiên, không bị ảnh hưởng bởi điểm số
    matching_courses = [item for item in recommendations['courses'] if getattr(item[0], 'grade_match', False)]
    non_matching_courses = [item for item in recommendations['courses'] if not getattr(item[0], 'grade_match', False)]
    
    matching_tutors = [item for item in recommendations['tutors'] if getattr(item[0], 'grade_match', False)]
    non_matching_tutors = [item for item in recommendations['tutors'] if not getattr(item[0], 'grade_match', False)]
    
    matching_materials = [item for item in recommendations['materials'] if getattr(item[0], 'grade_match', False)]
    non_matching_materials = [item for item in recommendations['materials'] if not getattr(item[0], 'grade_match', False)]
    
    # Chia các nhóm theo cả khối lớp và môn học yêu thích
    matching_grade_subject_courses = [item for item in matching_courses if getattr(item[0], 'subject_match', False)]
    matching_grade_only_courses = [item for item in matching_courses if not getattr(item[0], 'subject_match', False)]
    non_matching_grade_subject_courses = [item for item in non_matching_courses if getattr(item[0], 'subject_match', False)]
    non_matching_grade_or_subject_courses = [item for item in non_matching_courses if not getattr(item[0], 'subject_match', False)]
    
    matching_grade_subject_tutors = [item for item in matching_tutors if getattr(item[0], 'subject_match', False)]
    matching_grade_only_tutors = [item for item in matching_tutors if not getattr(item[0], 'subject_match', False)]
    non_matching_grade_subject_tutors = [item for item in non_matching_tutors if getattr(item[0], 'subject_match', False)]
    non_matching_grade_or_subject_tutors = [item for item in non_matching_tutors if not getattr(item[0], 'subject_match', False)]
    
    matching_grade_subject_materials = [item for item in matching_materials if getattr(item[0], 'subject_match', False)]
    matching_grade_only_materials = [item for item in matching_materials if not getattr(item[0], 'subject_match', False)]
    non_matching_grade_subject_materials = [item for item in non_matching_materials if getattr(item[0], 'subject_match', False)]
    non_matching_grade_or_subject_materials = [item for item in non_matching_materials if not getattr(item[0], 'subject_match', False)]
    
    # Sắp xếp các nhóm riêng biệt dựa trên điểm số 
    matching_grade_subject_courses.sort(key=lambda x: -x[1])
    matching_grade_only_courses.sort(key=lambda x: -x[1])
    non_matching_grade_subject_courses.sort(key=lambda x: -x[1])
    non_matching_grade_or_subject_courses.sort(key=lambda x: -x[1])
    
    matching_grade_subject_tutors.sort(key=lambda x: -x[1])
    matching_grade_only_tutors.sort(key=lambda x: -x[1])
    non_matching_grade_subject_tutors.sort(key=lambda x: -x[1])
    non_matching_grade_or_subject_tutors.sort(key=lambda x: -x[1])
    
    matching_grade_subject_materials.sort(key=lambda x: -x[1])
    matching_grade_only_materials.sort(key=lambda x: -x[1])
    non_matching_grade_subject_materials.sort(key=lambda x: -x[1])
    non_matching_grade_or_subject_materials.sort(key=lambda x: -x[1])
    
    # Kết hợp lại với thứ tự ưu tiên: khớp cả khối lớp và môn học > khớp khối lớp > khớp môn học > không khớp gì
    recommendations['courses'] = matching_grade_subject_courses + matching_grade_only_courses + non_matching_grade_subject_courses + non_matching_grade_or_subject_courses
    recommendations['tutors'] = matching_grade_subject_tutors + matching_grade_only_tutors + non_matching_grade_subject_tutors + non_matching_grade_or_subject_tutors
    recommendations['materials'] = matching_grade_subject_materials + matching_grade_only_materials + non_matching_grade_subject_materials + non_matching_grade_or_subject_materials
    
    # In ra thông tin debug để kiểm tra
    print(f"Số khóa học khớp cả khối lớp và môn học: {len(matching_grade_subject_courses)}")
    print(f"Số khóa học chỉ khớp khối lớp: {len(matching_grade_only_courses)}")
    print(f"Số khóa học chỉ khớp môn học: {len(non_matching_grade_subject_courses)}")
    print(f"Số khóa học không khớp gì: {len(non_matching_grade_or_subject_courses)}")
    
    print(f"Số gia sư khớp cả khối lớp và môn học: {len(matching_grade_subject_tutors)}")
    print(f"Số gia sư chỉ khớp khối lớp: {len(matching_grade_only_tutors)}")
    print(f"Số gia sư chỉ khớp môn học: {len(non_matching_grade_subject_tutors)}")
    print(f"Số gia sư không khớp gì: {len(non_matching_grade_or_subject_tutors)}")
    
    print(f"Số tài liệu khớp cả khối lớp và môn học: {len(matching_grade_subject_materials)}")
    print(f"Số tài liệu chỉ khớp khối lớp: {len(matching_grade_only_materials)}")
    print(f"Số tài liệu chỉ khớp môn học: {len(non_matching_grade_subject_materials)}")
    print(f"Số tài liệu không khớp gì: {len(non_matching_grade_or_subject_materials)}")
    
    # Hiển thị một số mẫu để kiểm tra
    if matching_grade_subject_courses:
        print("\nMột số khóa học khớp cả khối lớp và môn học:")
        for i, (course, score) in enumerate(matching_grade_subject_courses[:3]):
            print(f"  {i+1}. {course.name} - Grade: {course.grade_level}, Subject: {course.subject}, Score: {score}")
            
    # Remove scores from the final lists
    recommendations['courses'] = [item[0] for item in recommendations['courses']]
    recommendations['tutors'] = [item[0] for item in recommendations['tutors']]
    recommendations['materials'] = [item[0] for item in recommendations['materials']]
    
    if request.method == 'POST':
        # For AJAX requests, return only the recommendations container
        return render_template('_recommendations.html', 
                             recommendations=recommendations,
                             active_tab=active_tab)
    
    return render_template('recommendations.html', 
                         recommendations=recommendations,
                         active_tab=active_tab,
                         all_subjects=sorted(all_subjects),
                         all_grades=sorted(all_grades),
                         all_methods=sorted(all_methods))

def apply_filters(item, filters):
    """Apply filters to an item based on its type"""
    # Chuẩn hóa tên môn học để so sánh
    def normalize_subject(subject):
        subject = subject.strip().lower()
        if "hoá" in subject:
            subject = subject.replace("hoá", "hóa")
        return subject.title()
    
    # Tách nhiều khối lớp nếu có
    def split_grades(grade_level):
        if not grade_level:
            return []
        if "," in grade_level:
            return [g.strip() for g in grade_level.split(",")]
        return [grade_level.strip()]
    
    # Chuẩn hóa định dạng khối lớp
    def normalize_grade(grade):
        if not grade:
            return ""
        grade = grade.strip().lower()
        # Trích xuất số lớp từ chuỗi
        if 'lớp' in grade:
            # Nếu chứa "lớp", chỉ lấy phần số
            grade_num = ''.join(filter(str.isdigit, grade))
            if grade_num:
                return grade_num
        # Nếu đã là số, trả về nguyên dạng
        if grade.isdigit():
            return grade
        return grade  # Trả về nguyên bản nếu không thể chuẩn hóa
    
    # Kiểm tra môn học
    if filters['subject']:
        normalized_filter_subject = normalize_subject(filters['subject'])
        normalized_item_subject = normalize_subject(item.subject)
        if normalized_item_subject != normalized_filter_subject:
            return False
        
    # Kiểm tra khối lớp
    if filters['grade']:
        item_grades = []
        if isinstance(item, Course):
            item_grades = split_grades(item.grade_level)
        elif isinstance(item, Tutor):
            item_grades = split_grades(item.specialized_grade)
        elif isinstance(item, Material):
            item_grades = split_grades(item.grade_level)
            
        # Chuẩn hóa khối lớp của filter và item để so sánh
        normalized_filter_grade = normalize_grade(filters['grade'])
        normalized_item_grades = [normalize_grade(g) for g in item_grades]
        
        # Kiểm tra xem khối lớp được lọc có nằm trong danh sách khối lớp của item không
        if normalized_filter_grade not in normalized_item_grades:
            return False
            
    # Kiểm tra phương pháp học
    if filters['method']:
        if isinstance(item, Course) and item.teaching_method != filters['method']:
            return False
        elif isinstance(item, Tutor) and item.teaching_method != filters['method']:
            return False
            
    return True

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
                flash('Thiếu thông tin trung tâm hoặc thời gian học')
                return redirect(url_for('study_plan'))

            # Kiểm tra tính hợp lệ của định dạng thời gian với nhiều định dạng khác nhau
            time_slot_patterns = [
                r'^\d{1,2}h\d{0,2}-\d{1,2}h\d{0,2} thứ [2-7]$',  # Định dạng chuẩn VD: 7h45-10h10 thứ 3
                r'^\d{1,2}h\d{0,2} thứ [2-7]$',                 # Định dạng ngắn VD: 18h thứ 3
                r'^\d{1,2}:\d{0,2}-\d{1,2}:\d{0,2} thứ [2-7]$',  # Định dạng với dấu : VD: 7:45-10:10 thứ 3
                r'^\d{1,2}:\d{0,2} thứ [2-7]$',                 # Định dạng ngắn với dấu : VD: 18:00 thứ 3
                r'^.*$'                                         # Chấp nhận tất cả định dạng khác tạm thời
            ]
            
            valid_format = False
            for pattern in time_slot_patterns:
                if re.match(pattern, time_slot):
                    valid_format = True
                    break
                    
            if not valid_format:
                flash('Định dạng thời gian không hợp lệ. Định dạng yêu cầu: "7h45-10h10 thứ 3"')
                return redirect(url_for('study_plan'))

            # Check for time slot conflicts
            if check_time_conflict(study_plan, time_slot):
                flash('Thời gian học bị trùng với lịch học hiện có')
                return redirect(url_for('study_plan'))

            # Kiểm tra xem course đã có trong selected_courses chưa
            existing_course = SelectedCourse.query.filter_by(
                study_plan_id=study_plan.id,
                course_id=course_id
            ).first()
            
            if existing_course:
                flash('Trung tâm này đã được thêm vào lịch học')
                return redirect(url_for('study_plan'))

            selected_course = SelectedCourse(
                study_plan_id=study_plan.id,
                course_id=course_id,
                time_slot=time_slot
            )
            db.session.add(selected_course)
            db.session.commit()
            flash('Đã thêm trung tâm vào kế hoạch học tập')

        elif action == 'add_tutor':
            tutor_id = request.form.get('tutor_id')
            selected_time_slot = request.form.get('selected_time_slot')
            if not (tutor_id and selected_time_slot):
                flash('Thiếu thông tin gia sư hoặc thời gian học')
                return redirect(url_for('study_plan'))

            # Kiểm tra tính hợp lệ của định dạng thời gian
            time_slot_patterns = [
                r'^\d{1,2}h\d{0,2} thứ [2-7]$',                 # Định dạng chuẩn VD: 18h thứ 3
                r'^\d{1,2}:\d{0,2} thứ [2-7]$',                 # Định dạng với dấu : VD: 18:00 thứ 3
                r'^.*$'                                         # Chấp nhận tất cả định dạng khác tạm thời
            ]
            
            valid_format = False
            for pattern in time_slot_patterns:
                if re.match(pattern, selected_time_slot):
                    valid_format = True
                    break
                    
            if not valid_format:
                flash('Định dạng thời gian không hợp lệ. Định dạng yêu cầu: "18h thứ 3"')
                return redirect(url_for('study_plan'))

            # Validate against tutor's available time slots
            tutor = Tutor.query.get(tutor_id)
            if not tutor or not is_valid_tutor_time_slot(tutor.teaching_time, selected_time_slot):
                flash('Thời gian đã chọn không có sẵn cho gia sư này')
                return redirect(url_for('study_plan'))

            # Check for time slot conflicts
            if check_time_conflict(study_plan, selected_time_slot):
                flash('Thời gian học bị trùng với lịch học hiện có')
                return redirect(url_for('study_plan'))

            selected_tutor = SelectedTutor(
                study_plan_id=study_plan.id,
                tutor_id=tutor_id,
                selected_time_slot=selected_time_slot
            )
            db.session.add(selected_tutor)
            db.session.commit()
            flash('Đã thêm gia sư vào kế hoạch học tập')

        elif action == 'add_material':
            material_id = request.form.get('material_id')
            time_slots = request.form.get('time_slots')
            if not (material_id and time_slots):
                flash('Thiếu thông tin tài liệu hoặc thời gian học')
                return redirect(url_for('study_plan'))

            try:
                time_slots_list = json.loads(time_slots)
                # Kiểm tra tính hợp lệ của định dạng thời gian cho mỗi slot
                for slot in time_slots_list:
                    if not slot or not slot.strip():
                        continue
                        
                    time_slot_patterns = [
                        r'^\d{1,2}h\d{0,2}-\d{1,2}h\d{0,2} thứ [2-7]$',  # Định dạng chuẩn VD: 7h45-10h10 thứ 3
                        r'^\d{1,2}h\d{0,2} thứ [2-7]$',                 # Định dạng ngắn VD: 18h thứ 3
                        r'^\d{1,2}:\d{0,2}-\d{1,2}:\d{0,2} thứ [2-7]$',  # Định dạng với dấu : VD: 7:45-10:10 thứ 3
                        r'^\d{1,2}:\d{0,2} thứ [2-7]$',                 # Định dạng ngắn với dấu : VD: 18:00 thứ 3
                        r'^.*$'                                         # Chấp nhận tất cả định dạng khác tạm thời
                    ]
                    
                    valid_format = False
                    for pattern in time_slot_patterns:
                        if re.match(pattern, slot):
                            valid_format = True
                            break
                            
                    if not valid_format:
                        flash(f'Định dạng thời gian không hợp lệ: {slot}. Định dạng yêu cầu: "7h-9h thứ 3"')
                        return redirect(url_for('study_plan'))

                    # Kiểm tra xung đột thời gian
                    if check_time_conflict(study_plan, slot):
                        flash(f'Thời gian học bị trùng với lịch học hiện có: {slot}')
                        return redirect(url_for('study_plan'))

                # Chỉ lưu các slot không trống
                filtered_slots = [slot for slot in time_slots_list if slot and slot.strip()]
                if not filtered_slots:
                    flash('Không có thời gian hợp lệ được nhập vào')
                    return redirect(url_for('study_plan'))
                
                selected_material = SelectedMaterial(
                    study_plan_id=study_plan.id,
                    material_id=material_id,
                    time_slots=json.dumps(filtered_slots)
                )
                db.session.add(selected_material)
                db.session.commit()
                flash('Đã thêm tài liệu vào kế hoạch học tập')
            except json.JSONDecodeError:
                flash('Định dạng thời gian không hợp lệ')
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
                flash('Đã xóa mục khỏi lịch học')
                
        elif action == 'remove_study_plan_item':
            study_plan_item_id = request.form.get('study_plan_item_id')
            if study_plan_item_id:
                # Remove the item from study plan items
                item = StudyPlanItem.query.filter_by(id=study_plan_item_id, user_id=current_user.id).first()
                if item:
                    db.session.delete(item)
                    db.session.commit()
                    flash('Đã xóa mục khỏi kế hoạch học tập')

        return redirect(url_for('study_plan'))

    # Get all selected items
    selected_courses = study_plan.selected_courses
    selected_tutors = study_plan.selected_tutors
    selected_materials = study_plan.selected_materials

    # Get all available items for selection
    courses = Course.query.all()
    tutors = Tutor.query.all()
    materials = Material.query.all()
    
    # Get study plan items
    study_plan_items = StudyPlanItem.query.filter_by(user_id=current_user.id).all()

    # Sort items by time
    sorted_items = sort_items_by_time(selected_courses, selected_tutors, selected_materials)
    
    # Debug info
    print(f"Sorted items: {sorted_items}")
    for item in sorted_items:
        print(f"Item: type={item['type']}, day={item['day']} ({type(item['day'])}), start={item['start']}, end={item['end']}")

    return render_template('study_plan.html',
                         study_plan=study_plan,
                         selected_courses=selected_courses,
                         selected_tutors=selected_tutors,
                         selected_materials=selected_materials,
                         courses=courses,
                         tutors=tutors,
                         materials=materials,
                         study_plan_items=study_plan_items,
                         sorted_items=sorted_items)

@app.route('/study_plan/add', methods=['POST'])
@login_required
def add_to_study_plan():
    try:
        item_type = request.form.get('type')
        item_id = request.form.get('id')
        name = request.form.get('name')
        subject = request.form.get('subject')
        grade = request.form.get('grade')
        method = request.form.get('method')
        schedule = request.form.get('schedule')  # Lấy thông tin lịch từ request
        
        # Lấy lịch học từ Course nếu item_type là 'course'
        if item_type == 'course':
            course = Course.query.get(item_id)
            if course and course.teaching_time:
                # Ưu tiên sử dụng lịch của trung tâm
                schedule = course.teaching_time
                print(f"Sử dụng lịch học của trung tâm: {schedule}")

        # Kiểm tra xem item đã tồn tại trong study plan chưa
        existing_item = StudyPlanItem.query.filter_by(
            user_id=current_user.id,
            item_type=item_type,
            item_id=item_id
        ).first()
        
        if existing_item:
            return jsonify({'success': False, 'message': 'Đề xuất này đã có trong kế hoạch học tập của bạn'})

        # Create a new study plan item
        study_plan_item = StudyPlanItem(
            user_id=current_user.id,
            item_type=item_type,
            item_id=item_id,
            name=name,
            subject=subject,
            grade=grade,
            method=method,
            time_slots=schedule  # Lưu thông tin lịch của trung tâm nếu có
        )
        
        db.session.add(study_plan_item)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Đã thêm vào kế hoạch học tập thành công'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Lỗi: {str(e)}'})

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

        # Nếu chuỗi trống, trả về None
        if not time_slot or not time_slot.strip():
            return None, None, None

        # Xử lý các định dạng khác nhau - tách ngày và thời gian
        time_parts = time_slot.lower().split('thứ')
        if len(time_parts) < 2:
            # Không có phần "thứ" - định dạng không đúng
            return None, None, None
            
        time_part = time_parts[0].strip()
        day_part = time_parts[1].strip()
        
        # Xử lý phần ngày
        day = day_map.get(f'thứ {day_part}')
        if not day and day_part.isdigit():
            day = int(day_part)
            if day < 2 or day > 7:
                return None, None, None
        if not day:
            return None, None, None

        # Xử lý phần thời gian
        if '-' in time_part:
            # Khoảng thời gian (7h45-10h10)
            start, end = time_part.split('-')
            start_minutes = time_to_minutes(start.strip())
            end_minutes = time_to_minutes(end.strip())
        else:
            # Chỉ có thời gian bắt đầu (18h)
            start_minutes = time_to_minutes(time_part.strip())
            end_minutes = start_minutes + 60  # Giả sử kéo dài 1 giờ
            
        return start_minutes, end_minutes, day
    except Exception as e:
        print(f"Lỗi xử lý định dạng thời gian: {e}")
        return None, None, None

def time_to_minutes(time_str):
    """Convert time string (e.g., '7h45' or '7:45') to minutes since midnight."""
    # Chuẩn hóa định dạng thời gian
    time_str = time_str.replace('h', ':').strip()
    if ':' not in time_str:
        # Chỉ có giờ, không có phút
        time_str = f"{time_str}:00"
        
    try:
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    except Exception as e:
        print(f"Lỗi chuyển đổi thời gian sang phút: {e}")
        return 0  # Mặc định là 0 phút từ 0h

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
    day_names = {2: "Hai", 3: "Ba", 4: "Tư", 5: "Năm", 6: "Sáu", 7: "Bảy"}

    # Add courses
    for course in selected_courses:
        start, end, day = parse_time_slot(course.time_slot)
        if start is not None:
            items.append({
                'type': 'course',
                'item': course,
                'start': start,
                'end': end,
                'day': day,  # Đảm bảo day là số nguyên (2-7)
                'day_name': day_names.get(day, "")
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
                'day': day,  # Đảm bảo day là số nguyên (2-7)
                'day_name': day_names.get(day, "")
            })

    # Add materials
    for material in selected_materials:
        try:
            time_slots = json.loads(material.time_slots)
            for slot in time_slots:
                start, end, day = parse_time_slot(slot)
                if start is not None:
                    items.append({
                        'type': 'material',
                        'item': material,
                        'start': start,
                        'end': end,
                        'day': day,  # Đảm bảo day là số nguyên (2-7)
                        'day_name': day_names.get(day, "")
                    })
        except json.JSONDecodeError:
            print(f"Lỗi phân tích thời gian cho tài liệu {material.id}: {material.time_slots}")
            continue

    # Sort by day, start time, and type (course > tutor > material)
    return sorted(items, key=lambda x: (x['day'], x['start'], {'course': 0, 'tutor': 1, 'material': 2}[x['type']]))

@app.route('/api/tutor/<int:tutor_id>/time-slots')
@login_required
def get_tutor_time_slots(tutor_id):
    tutor = Tutor.query.get_or_404(tutor_id)
    time_slots = []
    
    # In thông tin debug
    print(f"Gia sư ID: {tutor_id}, Thời gian dạy: {tutor.teaching_time}")
    
    # Extract available time slots from teaching_time field
    if tutor.teaching_time:
        time_slots = tutor.teaching_time.split(';')
        time_slots = [slot.strip() for slot in time_slots if slot.strip()]
        print(f"Các khung giờ đã trích xuất: {time_slots}")
    
    return jsonify({'time_slots': time_slots})

def init_db():
    with app.app_context():
        try:
            # Create tables if they don't exist
            db.create_all()
            print("Tables created successfully.")
            return True
        except OperationalError as e:
            print(f"Database error: {e}")
            return False

@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's study plan items
    study_plan_items = StudyPlanItem.query.filter_by(user_id=current_user.id).all()
    
    # Get recommendations count
    recommendations = {
        'courses': Course.query.count(),
        'tutors': Tutor.query.count(),
        'materials': Material.query.count()
    }
    
    return render_template('dashboard.html', 
                         study_plan_items=study_plan_items,
                         recommendations=recommendations)

@app.route('/check_data')
@login_required
def check_data():
    courses = Course.query.all()
    tutors = Tutor.query.all()
    materials = Material.query.all()
    
    return jsonify({
        'courses': len(courses),
        'tutors': len(tutors),
        'materials': len(materials),
        'sample_course': {
            'id': courses[0].id if courses else None,
            'name': courses[0].name if courses else None,
            'subject': courses[0].subject if courses else None,
            'grade_level': courses[0].grade_level if courses else None
        } if courses else None,
        'sample_tutor': {
            'id': tutors[0].id if tutors else None,
            'name': tutors[0].name if tutors else None,
            'subject': tutors[0].subject if tutors else None,
            'specialized_grade': tutors[0].specialized_grade if tutors else None
        } if tutors else None,
        'sample_material': {
            'id': materials[0].id if materials else None,
            'name': materials[0].name if materials else None,
            'subject': materials[0].subject if materials else None,
            'grade_level': materials[0].grade_level if materials else None
        } if materials else None
    })

@app.route('/add_test_data')
@login_required
def add_test_data():
    # Add test courses
    courses = [
        Course(
            name='Trung tâm Toán A',
            subject='Toán',
            grade_level='Lớp 10',
            teaching_method='Trực tiếp',
            teaching_time='7h45-10h10 thứ 3',
            location='Quận 1, TP.HCM'
        ),
        Course(
            name='Trung tâm Lý B',
            subject='Lý',
            grade_level='11',  # Định dạng chỉ có số
            teaching_method='Online',
            teaching_time='13h30-16h thứ 5',
            location='Quận 3, TP.HCM'
        ),
        Course(
            name='Trung tâm Hóa C',
            subject='Hóa',
            grade_level='Lớp 12',
            teaching_method='Hybrid',
            teaching_time='18h-20h30 thứ 7',
            location='Quận 5, TP.HCM'
        ),
        # Thêm nhiều trung tâm hơn
        Course(
            name='Trung tâm Toán D',
            subject='Toán',
            grade_level='10',  # Định dạng chỉ có số
            teaching_method='Online',
            teaching_time='18h-20h30 thứ 4',
            location='Quận 7, TP.HCM'
        ),
        Course(
            name='Trung tâm Lý E',
            subject='Lý',
            grade_level='Lớp11',  # Định dạng không có khoảng cách
            teaching_method='Trực tiếp',
            teaching_time='7h45-10h10 thứ 6',
            location='Quận 2, TP.HCM'
        ),
        Course(
            name='Trung tâm Hóa F',
            subject='Hóa',
            grade_level='12',  # Định dạng chỉ có số
            teaching_method='Hybrid',
            teaching_time='13h30-16h thứ 7',
            location='Quận 4, TP.HCM'
        )
    ]
    
    # Add test tutors
    tutors = [
        Tutor(
            name='Gia sư Toán X',
            subject='Toán',
            specialized_grade='10',  # Định dạng chỉ có số
            teaching_method='Trực tiếp',
            teaching_time='18h thứ 3;18h thứ 5',
            experience=5
        ),
        Tutor(
            name='Gia sư Lý Y',
            subject='Lý',
            specialized_grade='Lớp 11',
            teaching_method='Online',
            teaching_time='19h thứ 4;19h thứ 6',
            experience=3
        ),
        Tutor(
            name='Gia sư Hóa Z',
            subject='Hóa',
            specialized_grade='Lớp12',  # Định dạng không có khoảng cách
            teaching_method='Hybrid',
            teaching_time='20h thứ 3;20h thứ 5',
            experience=4
        ),
        # Thêm nhiều gia sư hơn
        Tutor(
            name='Gia sư Toán W',
            subject='Toán',
            specialized_grade='Lớp 10',
            teaching_method='Online',
            teaching_time='19h thứ 4;19h thứ 6',
            experience=6
        ),
        Tutor(
            name='Gia sư Lý V',
            subject='Lý',
            specialized_grade='11',  # Định dạng chỉ có số
            teaching_method='Trực tiếp',
            teaching_time='18h thứ 3;18h thứ 5',
            experience=4
        ),
        Tutor(
            name='Gia sư Hóa U',
            subject='Hóa',
            specialized_grade='Lớp 12',
            teaching_method='Hybrid',
            teaching_time='20h thứ 4;20h thứ 6',
            experience=5
        )
    ]
    
    # Add test materials
    materials = [
        Material(
            name='Sách Toán 10',
            subject='Toán',
            grade_level='10',  # Định dạng chỉ có số
            material_type='Sách',
            description='Sách giáo khoa và bài tập Toán 10'
        ),
        Material(
            name='Sách Lý 11',
            subject='Lý',
            grade_level='Lớp 11',
            material_type='Sách',
            description='Sách giáo khoa và bài tập Lý 11'
        ),
        Material(
            name='Sách Hóa 12',
            subject='Hóa',
            grade_level='Lớp12',  # Định dạng không có khoảng cách
            material_type='Sách',
            description='Sách giáo khoa và bài tập Hóa 12'
        ),
        # Thêm nhiều tài liệu hơn
        Material(
            name='Bài tập Toán 10',
            subject='Toán',
            grade_level='Lớp 10',
            material_type='Bài tập',
            description='Tuyển tập bài tập Toán 10'
        ),
        Material(
            name='Bài tập Lý 11',
            subject='Lý',
            grade_level='11',  # Định dạng chỉ có số
            material_type='Bài tập',
            description='Tuyển tập bài tập Lý 11'
        ),
        Material(
            name='Bài tập Hóa 12',
            subject='Hóa',
            grade_level='12',  # Định dạng chỉ có số
            material_type='Bài tập',
            description='Tuyển tập bài tập Hóa 12'
        )
    ]
    
    try:
        # Xóa dữ liệu cũ
        Course.query.delete()
        Tutor.query.delete()
        Material.query.delete()
        db.session.commit()
        
        # Thêm dữ liệu mới
        for course in courses:
            db.session.add(course)
        for tutor in tutors:
            db.session.add(tutor)
        for material in materials:
            db.session.add(material)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Test data added successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/feedback', methods=['GET', 'POST'])
@login_required
def feedback():
    success = False
    
    if request.method == 'POST':
        feedback_type = request.form.get('feedback_type')
        content = request.form.get('content')
        rating = request.form.get('rating')
        
        if feedback_type and content and rating:
            new_feedback = Feedback(
                user_id=current_user.id,
                feedback_type=feedback_type,
                content=content,
                rating=int(rating)
            )
            db.session.add(new_feedback)
            db.session.commit()
            success = True
            flash('Cảm ơn bạn đã gửi phản hồi!')
        else:
            flash('Vui lòng điền đầy đủ thông tin phản hồi')
    
    # Get user's previous feedback
    feedbacks = Feedback.query.filter_by(user_id=current_user.id).order_by(Feedback.created_at.desc()).all()
    
    return render_template('feedback.html', success=success, feedbacks=feedbacks)

if __name__ == '__main__':
    with app.app_context():
        init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)