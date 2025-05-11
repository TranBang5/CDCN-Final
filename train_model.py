import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
from datetime import datetime
from models.recommender import RecommendationModel, load_and_preprocess_data
import os
import matplotlib.pyplot as plt

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
            missing_values = df.isnull().sum()
            if missing_values.any():
                print(f"Missing values in {df_name}:")
                print(missing_values[missing_values > 0])
                for col in df.columns:
                    if df[col].dtype in ['object', 'string']:
                        df[col].fillna('unknown', inplace=True)
                    else:
                        df[col].fillna(0, inplace=True)
            
            numeric_columns = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns
            for col in numeric_columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                df[col] = df[col].replace(0, 0.001)
            
            print(f"{df_name} shape: {df.shape}")
            print(f"{df_name} dtypes:\n{df.dtypes}")
            
            categorical_columns = [
                'Trường học hiện tại', 'Khối Lớp hiện tại', 'Mục tiêu học',
                'Môn học yêu thích', 'Phương pháp học yêu thích', 'Tên Trung Tâm',
                'Môn học', 'Khối Lớp', 'Phương pháp học', 'Tên gia sư',
                'Thời gian dạy học', 'Tên tài liệu', 'Loại tài liệu', 'Địa chỉ'
            ]
            for col in categorical_columns:
                if col in df.columns:
                    non_string = df[col][~df[col].apply(lambda x: isinstance(x, str))]
                    if not non_string.empty:
                        print(f"Non-string values in {df_name}.{col}: {non_string.tolist()}")
                    print(f"Sample values in {df_name}.{col}: {df[col].head(5).tolist()}")
    
    # Validate vocabulary consistency
    print("\nValidating vocabulary consistency...")
    student_cols = ['Trường học hiện tại', 'Khối Lớp hiện tại', 'Mục tiêu học', 'Môn học yêu thích', 'Phương pháp học yêu thích']
    for col in student_cols:
        train_values = set(data['student_course_train'][col].unique())
        student_values = set(data['student_data'][col].unique())
        missing_in_student = train_values - student_values
        if missing_in_student:
            print(f"Warning: Values in student_course_train.{col} not in student_data.{col}: {missing_in_student}")

def plot_history(history):
    for dataset in ['course', 'tutor', 'material']:
        hist = history[f'history_{dataset}']
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(hist['loss'], label='Loss')
        plt.title(f'{dataset.capitalize()} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(hist.get('factorized_top_k/top_1_categorical_accuracy', []), label='Top-1 Accuracy')
        plt.plot(hist.get('factorized_top_k/top_5_categorical_accuracy', []), label='Top-5 Accuracy')
        plt.plot(hist.get('factorized_top_k/top_10_categorical_accuracy', []), label='Top-10 Accuracy')
        plt.title(f'{dataset.capitalize()} Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{dataset}_training_plot.png')
        plt.close()

def train_and_evaluate():
    if os.environ.get('TF_ENABLE_ONEDNN_OPTS') == '0':
        print("oneDNN optimizations are disabled.")
    else:
        print("oneDNN optimizations are enabled. To disable, set TF_ENABLE_ONEDNN_OPTS=0.")

    print("Loading and preprocessing data...")
    data = load_and_preprocess_data()
    
    validate_data(data)
    
    student_features = data['student_data']
    course_features = data['course_data']
    tutor_features = data['tutor_data']
    material_features = data['material_data']
    student_course_train = data['student_course_train']
    student_tutor_train = data['student_tutor_train']
    student_material_train = data['student_material_train']
    student_course_test = data['student_course_test']
    student_tutor_test = data['student_tutor_test']
    student_material_test = data['student_material_test']
    
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
    
    print("Building model...")
    input_shape = {
        'ID Học Sinh': (None,),
        'Tên': (None,),
        'Trường học hiện tại': (None,),
        'Khối Lớp hiện tại': (None,),
        'Mục tiêu học': (None,),
        'Môn học yêu thích': (None,),
        'Phương pháp học yêu thích': (None,),
        'ID Trung Tâm': (None,),
        'Tên Trung Tâm': (None,),
        'Môn học': (None,),
        'Khối Lớp': (None,),
        'Phương pháp học': (None,),
        'Thời gian': (None,),
        'Chi phí': (None,),
        'Địa chỉ': (None,),
        'Đánh giá': (None,),
        'ID Gia Sư': (None,),
        'Tên gia sư': (None,),
        'Thời gian dạy học': (None,),
        'Kinh nghiệm giảng dạy': (None,),
        'ID Tài Liệu': (None,),
        'Tên tài liệu': (None,),
        'Loại tài liệu': (None,)
    }
    print("Input shapes defined:", input_shape)
    
    try:
        model.build(input_shape)
        print("Model built successfully. Summary:")
        model.summary()
    except Exception as e:
        print(f"Error building model: {str(e)}")
        raise
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0005,
            decay_steps=1000,
            decay_rate=0.9
        )
    )
    model.compile(optimizer=optimizer)
    
    batch_size = 32
    course_train_dataset = data['student_course_train_dataset'].batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    tutor_train_dataset = data['student_tutor_train_dataset'].batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    material_train_dataset = data['student_material_train_dataset'].batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    
    course_test_dataset = tf.data.Dataset.from_tensor_slices(dict(student_course_test)).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    tutor_test_dataset = tf.data.Dataset.from_tensor_slices(dict(student_tutor_test)).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    material_test_dataset = tf.data.Dataset.from_tensor_slices(dict(student_material_test)).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    
    progress_callback = TrainingProgressCallback()
    
    try:
        epochs = 20
        history = {'history_course': {}, 'history_tutor': {}, 'history_material': {}}
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            print("\nTraining on Course Data...")
            progress_callback.set_dataset("Course")
            history_course = model.fit(
                course_train_dataset,
                epochs=1,
                callbacks=[progress_callback],
                verbose=0
            )
            for key, value in history_course.history.items():
                history['history_course'].setdefault(key, []).extend(value)
            
            print("\nTraining on Tutor Data...")
            progress_callback.set_dataset("Tutor")
            history_tutor = model.fit(
                tutor_train_dataset,
                epochs=1,
                callbacks=[progress_callback],
                verbose=0
            )
            for key, value in history_tutor.history.items():
                history['history_tutor'].setdefault(key, []).extend(value)
            
            print("\nTraining on Material Data...")
            progress_callback.set_dataset("Material")
            history_material = model.fit(
                material_train_dataset,
                epochs=1,
                callbacks=[progress_callback],
                verbose=0
            )
            for key, value in history_material.history.items():
                history['history_material'].setdefault(key, []).extend(value)
        
        print("\nPlotting training history...")
        plot_history(history)
        
        print("\nEvaluating on test datasets...")
        test_metrics = {}
        
        print("Evaluating on Course test dataset...")
        test_metrics['course'] = model.evaluate(course_test_dataset, return_dict=True)
        
        print("Evaluating on Tutor test dataset...")
        test_metrics['tutor'] = model.evaluate(tutor_test_dataset, return_dict=True)
        
        print("Evaluating on Material test dataset...")
        test_metrics['material'] = model.evaluate(material_test_dataset, return_dict=True)
        
        print("\nTest Metrics:")
        for dataset, metrics in test_metrics.items():
            print(f"{dataset.capitalize()} Test Metrics:", metrics)
        
        print("Verifying model build state...")
        if model.built:
            print("Model is built. Number of weights:", len(model.weights))
        else:
            raise ValueError("Model is not built before saving weights!")
        
        print("Saving model weights...")
        model.save_weights('recommendation_model_weights.weights.h5')
        print("Model weights saved successfully to recommendation_model_weights.weights.h5")
        
        return {
            'history_course': history['history_course'],
            'history_tutor': history['history_tutor'],
            'history_material': history['history_material'],
            'test_metrics': test_metrics
        }
    
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    print("Starting training process...")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"TensorFlow Recommenders version: {tfrs.__version__}")
    history = train_and_evaluate()
    print("Training process completed.")
    print("Final training history:", history)
