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
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to avoid compatibility issues
RUN pip install --upgrade pip

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with no-cache to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary application files
COPY app.py .
COPY models/ models/
COPY templates/ templates/
COPY load_data.py .
COPY start.sh .

# Create necessary directories
RUN mkdir -p data checkpoints

# Copy only necessary data files
COPY data/*.csv data/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=development
ENV TF_DISABLE_GPU=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Expose port
EXPOSE 5000

# Make start script executable
RUN chmod +x start.sh

CMD ["./start.sh"]
