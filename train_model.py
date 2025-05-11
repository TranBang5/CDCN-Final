import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
from datetime import datetime
from models.recommender import RecommendationModel, load_and_preprocess_data
import os

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.batch_start_time = None
        self.batch_count = 0

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = datetime.now()
        self.batch_count += 1
        print(f"Starting batch {self.batch_count} for {self.current_dataset}...")
        print(f"Current time: {self.batch_start_time.strftime('%H:%M:%S')}")

    def on_train_batch_end(self, batch, logs=None):
        batch_end_time = datetime.now()
        batch_duration = (batch_end_time - self.batch_start_time).total_seconds()
        print(f"Finished batch {self.batch_count} for {self.current_dataset}")
        print(f"Batch duration: {batch_duration:.2f} seconds")
        print(f"Loss: {logs.get('loss', 'N/A')}")
        print(f"Metrics: {', '.join([f'{k}: {v:.4f}' for k, v in logs.items() if k != 'loss']) or 'None'}")
        print("-" * 50)

    def set_dataset(self, dataset_name):
        self.current_dataset = dataset_name
        self.batch_count = 0

def validate_data(data):
    print("Validating data...")
    for df_name, df in data.items():
        if isinstance(df, pd.DataFrame):
            print(f"\nValidating {df_name}...")
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                print(f"Missing values in {df_name}:")
                print(missing_values[missing_values > 0])
                # Fill missing values with appropriate defaults
                for col in df.columns:
                    if df[col].dtype in ['object', 'string']:
                        df[col].fillna('unknown', inplace=True)
                    else:
                        df[col].fillna(0, inplace=True)
            
            # Replace zero values in numeric columns with small positive value
            numeric_columns = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns
            for col in numeric_columns:
                df[col] = df[col].replace(0, 0.001)
            
            print(f"{df_name} shape: {df.shape}")
            print(f"{df_name} dtypes:\n{df.dtypes}")

def train_and_evaluate():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data()
    
    # Validate data
    validate_data(data)
    
    # Extract data
    student_features = data['student_data']
    course_features = data['course_data']
    tutor_features = data['tutor_data']
    material_features = data['material_data']
    student_course_train = data['student_course_train']
    student_tutor_train = data['student_tutor_train']
    student_material_train = data['student_material_train']
    
    # Initialize model
    print("Initializing RecommendationModel...")
    model = RecommendationModel(
        student_features=student_features,
        course_features=course_features,
        tutor_features=tutor_features,
        material_features=material_features,
        student_course_train=student_course_train,
        student_tutor_train=student_tutor_train,
        student_material_train=student_material_train
    )
    
    # Build model with sample input shape
    print("Building model...")
    # Define input shape based on tutor dataset schema (most detailed from prior logs)
    input_shape = {
        'ID Học Sinh': tf.TensorSpec(shape=(None,), dtype=tf.int64),
        'Tên': tf.TensorSpec(shape=(None,), dtype=tf.string),
        'Trường học hiện tại': tf.TensorSpec(shape=(None,), dtype=tf.string),
        'Khối Lớp hiện tại': tf.TensorSpec(shape=(None,), dtype=tf.string),
        'Mục tiêu học': tf.TensorSpec(shape=(None,), dtype=tf.string),
        'Môn học yêu thích': tf.TensorSpec(shape=(None,), dtype=tf.string),
        'Phương pháp học yêu thích': tf.TensorSpec(shape=(None,), dtype=tf.string),
        'ID Gia Sư': tf.TensorSpec(shape=(None,), dtype=tf.int64),
        'Tên gia sư': tf.TensorSpec(shape=(None,), dtype=tf.string),
        'Môn học': tf.TensorSpec(shape=(None,), dtype=tf.string),
        'Thời gian dạy học': tf.TensorSpec(shape=(None,), dtype=tf.string),
        'Khối Lớp': tf.TensorSpec(shape=(None,), dtype=tf.string),
        'Kinh nghiệm giảng dạy': tf.TensorSpec(shape=(None,), dtype=tf.float32),
        'Đánh giá': tf.TensorSpec(shape=(None,), dtype=tf.int64)
    }
    model.build(input_shape)
    print("Model built successfully. Summary:")
    model.summary()
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    
    # Prepare datasets
    batch_size = 8
    course_train_dataset = model.course_train_dataset.batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)
    tutor_train_dataset = model.tutor_train_dataset.batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)
    material_train_dataset = model.material_train_dataset.batch(batch_size, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)
    
    # Training progress callback
    progress_callback = TrainingProgressCallback()
    
    try:
        # Train on course data
        print("\nTraining on Course Data...")
        progress_callback.set_dataset("Course")
        history_course = model.fit(
            course_train_dataset,
            epochs=5,
            callbacks=[progress_callback],
            verbose=0
        )
        print("Course training completed.")
        print("Course training history:", history_course.history)
        
        # Train on tutor data
        print("\nTraining on Tutor Data...")
        progress_callback.set_dataset("Tutor")
        history_tutor = model.fit(
            tutor_train_dataset,
            epochs=5,
            callbacks=[progress_callback],
            verbose=0
        )
        print("Tutor training completed.")
        print("Tutor training history:", history_tutor.history)
        
        # Train on material data
        print("\nTraining on Material Data...")
        progress_callback.set_dataset("Material")
        history_material = model.fit(
            material_train_dataset,
            epochs=5,
            callbacks=[progress_callback],
            verbose=0
        )
        print("Material training completed.")
        print("Material training history:", history_material.history)
        
        # Verify model build state before saving
        print("Verifying model build state...")
        if model.built:
            print("Model is built. Number of weights:", len(model.weights))
        else:
            raise ValueError("Model is not built before saving weights!")
        
        # Save the model
        print("Saving model weights...")
        model.save_weights('recommendation_model_weights.weights.h5')
        print("Model weights saved successfully to recommendation_model_weights.weights.h5")
        
        return {
            'history_course': history_course.history,
            'history_tutor': history_tutor.history,
            'history_material': history_material.history
        }
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    print("Starting training process...")
    history = train_and_evaluate()
    print("Training process completed.")
    print("Final training history:", history)
