import pandas as pd
from models.database import db, Course, Tutor, Material
from app import app
import os
import logging
import re

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_number_from_string(text):
    """Extract number from string like '60.000 VNĐ/Ca' or '20 năm'"""
    try:
        # Remove all non-numeric characters except decimal point and commas (used for thousands separator)
        number_str = re.sub(r'[^\d.,]', '', str(text))
        
        # Replace comma with dot if used as decimal separator
        if ',' in number_str and '.' in number_str:
            # Both comma and dot present, assume comma is thousand separator
            number_str = number_str.replace(',', '')
        elif ',' in number_str:
            # Only comma present, might be decimal separator
            number_str = number_str.replace(',', '.')
            
        return float(number_str) * 1000  # Nhân 1000 để chuyển đổi đúng định dạng (VD: 60 -> 60,000)
    except:
        return 0.0

def validate_course_data(row):
    """Validate course data before inserting into database"""
    try:
        # Check required fields
        required_fields = ['ID Trung Tâm', 'Tên Trung Tâm', 'Môn học', 'Khối Lớp', 'Thời gian', 'Địa chỉ', 'Phương pháp học', 'Chi phí']
        for field in required_fields:
            if pd.isna(row[field]) or str(row[field]).strip() == '':
                logger.error(f"Missing required field {field} for course ID {row['ID Trung Tâm']}")
                return False
            
        # Validate ID
        if not str(row['ID Trung Tâm']).isdigit():
            logger.error(f"Invalid course ID format: {row['ID Trung Tâm']}")
            return False
            
        # Validate cost
        cost = extract_number_from_string(row['Chi phí'])
        if cost <= 0:
            logger.error(f"Invalid cost value for course ID {row['ID Trung Tâm']}: {cost}")
            return False
            
        # Validate teaching time format (không bắt buộc, chỉ kiểm tra nếu có)
        if not pd.isna(row['Thời gian']) and str(row['Thời gian']).strip() != '':
            time_str = str(row['Thời gian']).strip()
            # Kiểm tra định dạng thời gian đơn giản
            if 'thứ' not in time_str.lower():
                logger.warning(f"Teaching time format may be invalid for course ID {row['ID Trung Tâm']}: {time_str}")
            
        return True
    except Exception as e:
        logger.error(f"Error validating course data: {str(e)}")
        return False

def validate_tutor_data(row):
    """Validate tutor data before inserting into database"""
    try:
        # Check required fields
        required_fields = ['ID Gia Sư', 'Tên gia sư', 'Môn học', 'Khối Lớp', 'Thời gian dạy học', 'Kinh nghiệm giảng dạy', 'Phương pháp dạy']
        for field in required_fields:
            if pd.isna(row[field]) or str(row[field]).strip() == '':
                logger.error(f"Missing required field {field} for tutor ID {row['ID Gia Sư']}")
                return False
            
        # Validate ID
        if not str(row['ID Gia Sư']).isdigit():
            logger.error(f"Invalid tutor ID format: {row['ID Gia Sư']}")
            return False
            
        # Validate teaching experience
        exp = extract_number_from_string(row['Kinh nghiệm giảng dạy'])
        if exp < 0:
            logger.error(f"Invalid teaching experience for tutor ID {row['ID Gia Sư']}: {exp}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating tutor data: {str(e)}")
        return False

def validate_material_data(row):
    """Validate material data before inserting into database"""
    try:
        # Check required fields
        required_fields = ['ID Tài Liệu', 'Tên tài liệu', 'Môn học', 'Khối Lớp', 'Loại tài liệu']
        for field in required_fields:
            if pd.isna(row[field]) or str(row[field]).strip() == '':
                logger.error(f"Missing required field {field} for material ID {row['ID Tài Liệu']}")
                return False
            
        # Validate ID
        if not str(row['ID Tài Liệu']).isdigit():
            logger.error(f"Invalid material ID format: {row['ID Tài Liệu']}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating material data: {str(e)}")
        return False

def load_data_to_db():
    logger.info("Starting data import...")
    
    # Check if data directory exists
    if not os.path.exists('data'):
        logger.error("Data directory not found!")
        return
        
    # Load course data
    logger.info("Loading course data from trung_tam.csv...")
    try:
        course_data = pd.read_csv('data/trung_tam.csv', encoding='utf-8')
        logger.debug(f"Course data shape: {course_data.shape}")
        
        # Check for duplicate IDs
        duplicate_ids = course_data['ID Trung Tâm'].duplicated()
        if duplicate_ids.any():
            logger.error(f"Found duplicate course IDs: {course_data[duplicate_ids]['ID Trung Tâm'].tolist()}")
            return
            
        for index, row in course_data.iterrows():
            if validate_course_data(row):
                try:
                    course = Course(
                        id=int(row['ID Trung Tâm']),
                        name=row['Tên Trung Tâm'],
                        subject=row['Môn học'],
                        grade_level=row['Khối Lớp'],
                        schedule=row['Thời gian'],
                        address=row['Địa chỉ'],
                        teaching_method=row['Phương pháp học'],
                        cost=extract_number_from_string(row['Chi phí']),
                        teaching_time=row['Thời gian'],
                        location=row['Địa chỉ']
                    )
                    db.session.add(course)
                    logger.debug(f"Added course ID {row['ID Trung Tâm']}: {row['Tên Trung Tâm']}")
                except Exception as e:
                    logger.error(f"Error adding course ID {row['ID Trung Tâm']}: {str(e)}")
    except Exception as e:
        logger.error(f"Error loading course data: {str(e)}")
        return
    
    # Load tutor data
    logger.info("Loading tutor data from gia_su.csv...")
    try:
        tutor_data = pd.read_csv('data/gia_su.csv', encoding='utf-8')
        logger.debug(f"Tutor data shape: {tutor_data.shape}")
        
        # Check for duplicate IDs
        duplicate_ids = tutor_data['ID Gia Sư'].duplicated()
        if duplicate_ids.any():
            logger.error(f"Found duplicate tutor IDs: {tutor_data[duplicate_ids]['ID Gia Sư'].tolist()}")
            return
            
        for index, row in tutor_data.iterrows():
            if validate_tutor_data(row):
                try:
                    tutor = Tutor(
                        id=int(row['ID Gia Sư']),
                        name=row['Tên gia sư'],
                        subject=row['Môn học'],
                        specialized_grade=row['Khối Lớp'],
                        schedule=row['Thời gian dạy học'],
                        teaching_experience=extract_number_from_string(row['Kinh nghiệm giảng dạy']),
                        teaching_method=row['Phương pháp dạy'],
                        teaching_time=row['Thời gian dạy học'],
                        experience=extract_number_from_string(row['Kinh nghiệm giảng dạy'])
                    )
                    db.session.add(tutor)
                    logger.debug(f"Added tutor ID {row['ID Gia Sư']}: {row['Tên gia sư']}")
                except Exception as e:
                    logger.error(f"Error adding tutor ID {row['ID Gia Sư']}: {str(e)}")
    except Exception as e:
        logger.error(f"Error loading tutor data: {str(e)}")
        return
    
    # Load material data
    logger.info("Loading material data from tai_lieu.csv...")
    try:
        material_data = pd.read_csv('data/tai_lieu.csv', encoding='utf-8')
        logger.debug(f"Material data shape: {material_data.shape}")
        
        # Check for duplicate IDs
        duplicate_ids = material_data['ID Tài Liệu'].duplicated()
        if duplicate_ids.any():
            logger.error(f"Found duplicate material IDs: {material_data[duplicate_ids]['ID Tài Liệu'].tolist()}")
            return
            
        for index, row in material_data.iterrows():
            if validate_material_data(row):
                try:
                    material = Material(
                        id=int(row['ID Tài Liệu']),
                        name=row['Tên tài liệu'],
                        subject=row['Môn học'],
                        grade_level=row['Khối Lớp'],
                        material_type=row['Loại tài liệu'],
                        description=row.get('Mô tả', 'Không có mô tả')
                    )
                    db.session.add(material)
                    logger.debug(f"Added material ID {row['ID Tài Liệu']}: {row['Tên tài liệu']}")
                except Exception as e:
                    logger.error(f"Error adding material ID {row['ID Tài Liệu']}: {str(e)}")
    except Exception as e:
        logger.error(f"Error loading material data: {str(e)}")
        return
    
    try:
        db.session.commit()
        logger.info("Data import completed successfully!")
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error during data import commit: {str(e)}")
        raise

if __name__ == '__main__':
    with app.app_context():
        logger.info("Clearing existing data...")
        try:
            Course.query.delete()
            Tutor.query.delete()
            Material.query.delete()
            db.session.commit()
            logger.info("Existing data cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing existing data: {str(e)}")
            db.session.rollback()
        
        load_data_to_db()