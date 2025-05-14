# Use Python 3.8 base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    default-libmysqlclient-dev \
    pkg-config \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to avoid compatibility issues
RUN pip install --upgrade pip

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with no-cache to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory if it doesn't exist
RUN mkdir -p data

# Copy all necessary data files
COPY data/*.csv data/

# Create checkpoints directory
RUN mkdir -p checkpoints

# Copy model weights and bruteforce data (if they exist)
COPY checkpoints/* checkpoints/ 2>/dev/null || true
COPY bruteforce_data.npz . 2>/dev/null || true

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=development

# Expose port
EXPOSE 5000

# Start script
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]