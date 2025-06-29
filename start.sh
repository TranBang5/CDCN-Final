#!/bin/bash

# Copy model files if they exist in /app (build context) to the current directory
[ -f /app/subject_pca_model.pkl ] && cp /app/subject_pca_model.pkl . || echo "subject_pca_model.pkl not found"
[ -f /app/grade_pca_model.pkl ] && cp /app/grade_pca_model.pkl . || echo "grade_pca_model.pkl not found"

# Wait for MySQL to be ready
echo "Waiting for MySQL to be ready..."
while ! nc -z db 3306; do
  sleep 1
done
echo "MySQL is ready!"

# Initialize database
echo "Initializing database..."
python -c "from app import app, init_db; app.app_context().push(); init_db()"

# Check if data needs to be loaded
echo "Checking if data needs to be loaded..."
python -c "
from app import app
from models.database import db, Course, Tutor, Material
with app.app_context():
    course_count = Course.query.count()
    tutor_count = Tutor.query.count()
    material_count = Material.query.count()
    if course_count == 0 and tutor_count == 0 and material_count == 0:
        print('Database is empty, loading data...')
        exit(1)
    else:
        print(f'Database already has data: {course_count} courses, {tutor_count} tutors, {material_count} materials')
        exit(0)
"

# Load data if database is empty
if [ $? -eq 1 ]; then
    echo "Loading data..."
    python load_data.py
    if [ $? -ne 0 ]; then
        echo "Error: Failed to load data"
        exit 1
    fi
    echo "Data loaded successfully"
else
    echo "Skipping data load as database already has data"
fi

# Start the Flask application
echo "Starting Flask application..."
python app.py 