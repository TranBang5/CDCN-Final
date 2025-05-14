import pandas as pd
from models.database import db, Course, Tutor, Material
from app import app
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data_to_db():
    logger.info("Starting data import...")
    
    # Load course data
    logger.info("Loading course data from trung_tam.csv...")
    course_data = pd.read_csv('data/trung_tam.csv', encoding='utf-8')
    logger.debug(f"Course data shape: {course_data.shape}")
    for index, row in course_data.iterrows():
        try:
            course = Course(
                id=int(row['ID Trung Tâm']),
                center_name=row['Tên Trung Tâm'],
                subject=row['Môn học'],
                grade_level=row['Khối Lớp'],
                schedule=row['Thời gian'],
                address=row['Địa chỉ'],
                teaching_method=row['Phương pháp học'],
                cost=float(row['Chi phí'])
            )
            db.session.add(course)
            logger.debug(f"Added course ID {row['ID Trung Tâm']}: {row['Tên Trung Tâm']}")
        except Exception as e:
            logger.error(f"Error adding course ID {row['ID Trung Tâm']}: {str(e)}")
    
    # Load tutor data
    logger.info("Loading tutor data from gia_su.csv...")
    tutor_data = pd.read_csv('data/gia_su.csv', encoding='utf-8')
    logger.debug(f"Tutor data shape: {tutor_data.shape}")
    for index, row in tutor_data.iterrows():
        try:
            tutor = Tutor(
                id=int(row['ID Gia Sư']),
                name=row['Tên gia sư'],
                subject=row['Môn học'],
                specialized_grade=row['Khối Lớp'],
                schedule=row['Thời gian dạy học'],
                teaching_experience=float(row['Kinh nghiệm giảng dạy']),
                teaching_method=row['Phương pháp dạy']
            )
            db.session.add(tutor)
            logger.debug(f"Added tutor ID {row['ID Gia Sư']}: {row['Tên gia sư']}")
        except Exception as e:
            logger.error(f"Error adding tutor ID {row['ID Gia Sư']}: {str(e)}")
    
    # Load material data
    logger.info("Loading material data from tai_lieu.csv...")
    material_data = pd.read_csv('data/tai_lieu.csv', encoding='utf-8')
    logger.debug(f"Material data shape: {material_data.shape}")
    for index, row in material_data.iterrows():
        try:
            material = Material(
                id=int(row['ID Tài Liệu']),
                name=row['Tên tài liệu'],
                subject=row['Môn học'],
                grade_level=row['Khối Lớp'],
                material_type=row['Loại tài liệu']
            )
            db.session.add(material)
            logger.debug(f"Added material ID {row['ID Tài Liệu']}: {row['Tên tài liệu']}")
        except Exception as e:
            logger.error(f"Error adding material ID {row['ID Tài Liệu']}: {str(e)}")
    
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