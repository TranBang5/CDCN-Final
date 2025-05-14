#!/bin/bash

# Wait for MySQL to be ready
echo "Waiting for MySQL to be ready..."
while ! nc -z db 3306; do
  sleep 1
done
echo "MySQL is ready!"

# Initialize database and load data
echo "Initializing database..."
python -c "from app import app, init_db; app.app_context().push(); init_db()"

echo "Loading data..."
python load_data.py

# Start the Flask application
echo "Starting Flask application..."
python app.py 