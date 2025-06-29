import os
import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.recommender import RecommendationModel, load_and_preprocess_data
import logging
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_performance_report(
    course_precision, course_recall, history_course,
    tutor_precision, tutor_recall, history_tutor,
    material_precision, material_recall, history_material,
    checkpoint_dir
):
    """
    Generate a text file summarizing the model's performance metrics.
    
    Args:
        course_precision (float): Precision@10 for course model.
        course_recall (float): Recall@10 for course model.
        history_course: Training history for course model.
        tutor_precision (float): Precision@10 for tutor model.
        tutor_recall (float): Recall@10 for tutor model.
        history_tutor: Training history for tutor model.
        material_precision (float): Precision@10 for material model.
        material_recall (float): Recall@10 for material model.
        history_material: Training history for material model.
        checkpoint_dir (str): Directory to save the report.
    """
    report_path = os.path.join(checkpoint_dir, 'performance_report.txt')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Calculate number of epochs completed
    course_epochs = len(history_course.history['loss'])
    tutor_epochs = len(history_tutor.history['loss'])
    material_epochs = len(history_material.history['loss'])
    
    # Get final training and validation loss
    course_final_loss = history_course.history['loss'][-1]
    course_final_val_loss = history_course.history.get('val_loss', [None])[-1]
    tutor_final_loss = history_tutor.history['loss'][-1]
    tutor_final_val_loss = history_tutor.history.get('val_loss', [None])[-1]
    material_final_loss = history_material.history['loss'][-1]
    material_final_val_loss = history_material.history.get('val_loss', [None])[-1]
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== Recommendation System Performance Report ===\n")
            f.write(f"Generated on: {timestamp}\n")
            f.write(f"Checkpoint Directory: {checkpoint_dir}\n\n")
            
            f.write("1. Course Model Performance\n")
            f.write(f"   Precision@10: {course_precision:.4f}\n")
            f.write(f"   Recall@10: {course_recall:.4f}\n")
            f.write(f"   Epochs Completed: {course_epochs}\n")
            f.write(f"   Final Training Loss: {course_final_loss:.4f}\n")
            if course_final_val_loss is not None:
                f.write(f"   Final Validation Loss: {course_final_val_loss:.4f}\n")
            f.write("\n")
            
            f.write("2. Tutor Model Performance\n")
            f.write(f"   Precision@10: {tutor_precision:.4f}\n")
            f.write(f"   Recall@10: {tutor_recall:.4f}\n")
            f.write(f"   Epochs Completed: {tutor_epochs}\n")
            f.write(f"   Final Training Loss: {tutor_final_loss:.4f}\n")
            if tutor_final_val_loss is not None:
                f.write(f"   Final Validation Loss: {tutor_final_val_loss:.4f}\n")
            f.write("\n")
            
            f.write("3. Material Model Performance\n")
            f.write(f"   Precision@10: {material_precision:.4f}\n")
            f.write(f"   Recall@10: {material_recall:.4f}\n")
            f.write(f"   Epochs Completed: {material_epochs}\n")
            f.write(f"   Final Training Loss: {material_final_loss:.4f}\n")
            if material_final_val_loss is not None:
                f.write(f"   Final Validation Loss: {material_final_val_loss:.4f}\n")
            f.write("\n")
            
            f.write("Notes:\n")
            f.write("- Precision@10 and Recall@10 are evaluated on the test datasets.\n")
            f.write("- Training was performed with early stopping (patience=3) and Adagrad optimizer.\n")
            f.write("- Models and BruteForce data are saved in the checkpoint directory.\n")
            f.write("=====================================\n")
        
        logger.info(f"Performance report saved to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save performance report: {str(e)}")

def train_and_evaluate():
    logger.info("Starting model training...")
    data = load_and_preprocess_data()
    
    # Extract data components
    student_data = data['student_data']
    course_data = data['course_data']
    tutor_data = data['tutor_data']
    material_data = data['material_data']
    student_course_train = data['student_course_train']
    student_course_test = data['student_course_test']
    student_tutor_train = data['student_tutor_train']
    student_tutor_test = data['student_tutor_test']
    student_material_train = data['student_material_train']
    student_material_test = data['student_material_test']
    subject_vocab = data['subject_vocab']
    grade_vocab = data['grade_vocab']
    material_type_vocab = data['material_type_vocab']
    teaching_time_vocab = data['teaching_time_vocab']
    
    logger.info(f"Dataset sizes: student_course_train={len(student_course_train)}, "
                f"student_tutor_train={len(student_tutor_train)}, "
                f"student_material_train={len(student_material_train)}")
    logger.info(f"Test dataset sizes: student_course_test={len(student_course_test)}, "
                f"student_tutor_test={len(student_tutor_test)}, "
                f"student_material_test={len(student_material_test)}")
    logger.info(f"Estimated batches per epoch (course): {int(np.ceil(len(student_course_train) / 32))}")

    # Validate dataset shapes and dtypes
    logger.info("Validating dataset dtypes and grade columns...")
    for df, name in [
        (student_data, 'student_data'),
        (course_data, 'course_data'),
        (tutor_data, 'tutor_data'),
        (material_data, 'material_data'),
        (student_course_train, 'student_course_train'),
        (student_tutor_train, 'student_tutor_train'),
        (student_material_train, 'student_material_train'),
        (student_course_test, 'student_course_test'),
        (student_tutor_test, 'student_tutor_test'),
        (student_material_test, 'student_material_test')
    ]:
        if df.empty:
            logger.error(f"{name} is empty")
            raise ValueError(f"Empty dataset: {name}")
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            logger.warning(f"{name} contains object dtype columns: {object_cols}")
            for col in object_cols:
                df[col] = df[col].astype(str).fillna('unknown')
        # Validate grade columns
        for col in ['khoi_lop_hien_tai', 'khoi_lop']:
            if col in df.columns:
                unique_grades = df[col].unique()
                logger.info(f"{name} - {col}: unique values = {unique_grades}")
                if len(unique_grades) <= 1:
                    logger.warning(f"{name} - {col} has insufficient unique values: {unique_grades}")
    
    # Debug dataset iteration
    logger.info("Testing dataset iteration...")
    for dataset, name in [
        (data['student_course_train_dataset'], 'course_train'),
        (data['student_tutor_train_dataset'], 'tutor_train'),
        (data['student_material_train_dataset'], 'material_train'),
        (data['student_course_test_dataset'], 'course_test'),
        (data['student_tutor_test_dataset'], 'tutor_test'),
        (data['student_material_test_dataset'], 'material_test')
    ]:
        for batch in dataset.take(1):
            logger.info(f"{name} batch keys: {list(batch.keys())}")
            logger.info(f"{name} batch shapes: {{k: v.shape for k, v in batch.items()}}")

    # Prepare datasets
    course_train_dataset = data['student_course_train_dataset'].batch(64).cache().prefetch(tf.data.AUTOTUNE)
    course_test_dataset = data['student_course_test_dataset'].batch(64).cache().prefetch(tf.data.AUTOTUNE)
    tutor_train_dataset = data['student_tutor_train_dataset'].batch(64).cache().prefetch(tf.data.AUTOTUNE)
    tutor_test_dataset = data['student_tutor_test_dataset'].batch(64).cache().prefetch(tf.data.AUTOTUNE)
    material_train_dataset = data['student_material_train_dataset'].batch(64).cache().prefetch(tf.data.AUTOTUNE)
    material_test_dataset = data['student_material_test_dataset'].batch(64).cache().prefetch(tf.data.AUTOTUNE)
    
    logger.info("Creating RecommendationModel...")
    try:
        model = RecommendationModel(
            student_features=student_data,
            course_features=course_data,
            tutor_features=tutor_data,
            material_features=material_data,
            student_course_train=student_course_train,
            student_tutor_train=student_tutor_train,
            student_material_train=student_material_train,
            subject_vocab=subject_vocab,
            grade_vocab=grade_vocab,
            material_type_vocab=material_type_vocab,
            teaching_time_vocab=teaching_time_vocab
        )
    except Exception as e:
        logger.error(f"Failed to initialize RecommendationModel: {str(e)}")
        raise
    
    # Build the model by calling it with sample data
    logger.info("Building model with sample data...")
    for batch in course_train_dataset.take(1):
        _ = model(batch, training=False)
    for batch in tutor_train_dataset.take(1):
        _ = model(batch, training=False)
    for batch in material_train_dataset.take(1):
        _ = model(batch, training=False)
    
    logger.info("Compiling model...")
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.005))
    
    # Setup and validate checkpoint directory
    checkpoint_dir = os.path.normpath(os.path.abspath(os.path.join(os.getcwd(), 'checkpoints')))
    try:
        if os.path.exists(checkpoint_dir) and not os.path.isdir(checkpoint_dir):
            logger.warning(f"Removing file named 'checkpoints' at {checkpoint_dir}")
            os.remove(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        test_file = os.path.join(checkpoint_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        logger.info(f"Checkpoint directory created/verified at: {checkpoint_dir}")
    except Exception as e:
        logger.error(f"Failed to create or verify checkpoint directory: {str(e)}")
        checkpoint_dir = os.path.join(os.getcwd(), 'temp_checkpoints')
        logger.warning(f"Falling back to temporary checkpoint directory: {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup model saving paths
    model_save_path = os.path.join(checkpoint_dir, 'saved_model')
    student_model_save_path = os.path.join(checkpoint_dir, 'student_model')
    course_model_save_path = os.path.join(checkpoint_dir, 'course_model')
    tutor_model_save_path = os.path.join(checkpoint_dir, 'tutor_model')
    material_model_save_path = os.path.join(checkpoint_dir, 'material_model')
    
    # Setup checkpointing
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_{epoch}.weights.h5')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=False,
        monitor='loss',
        verbose=1,
        save_freq='epoch'
    )
    
    class TrainingProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logger.info(f"Epoch {epoch + 1} completed")
            for key, value in logs.items():
                logger.info(f"{key}: {value:.4f}")
    
    progress_callback = TrainingProgressCallback()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=3,
        restore_best_weights=True
    )
    
    epochs = 50
    
    logger.info("Training Course Model...")
    try:
        history_course = model.fit(
            course_train_dataset,
            validation_data=course_test_dataset,
            epochs=epochs,
            callbacks=[progress_callback, early_stopping, checkpoint_callback],
            verbose=1
        )
        logger.info("Course model training completed")
    except Exception as e:
        logger.error(f"Error training course model: {str(e)}")
        raise
    
    logger.info("Training Tutor Model...")
    try:
        history_tutor = model.fit(
            tutor_train_dataset,
            validation_data=tutor_test_dataset,
            epochs=epochs,
            callbacks=[progress_callback, early_stopping, checkpoint_callback],
            verbose=1
        )
        logger.info("Tutor model training completed")
    except Exception as e:
        logger.error(f"Error training tutor model: {str(e)}")
        raise
    
    logger.info("Training Material Model...")
    try:
        history_material = model.fit(
            material_train_dataset,
            validation_data=material_test_dataset,
            epochs=epochs,
            callbacks=[progress_callback, early_stopping, checkpoint_callback],
            verbose=1
        )
        logger.info("Material model training completed")
    except Exception as e:
        logger.error(f"Error training material model: {str(e)}")
        raise
    
    # Evaluate precision and recall on test datasets
    logger.info("Evaluating precision and recall on test datasets...")
    course_precision, course_recall = model.evaluate_test(
        course_test_dataset, 'course', 'id_trung_tam', model.course_model.candidate_embeddings['identifiers']
    )
    tutor_precision, tutor_recall = model.evaluate_test(
        tutor_test_dataset, 'tutor', 'id_gia_su', model.tutor_model.candidate_embeddings['identifiers']
    )
    material_precision, material_recall = model.evaluate_test(
        material_test_dataset, 'material', 'id_tai_lieu', model.material_model.candidate_embeddings['identifiers']
    )
    
    # Generate performance report
    logger.info("Generating performance report...")
    generate_performance_report(
        course_precision, course_recall, history_course,
        tutor_precision, tutor_recall, history_tutor,
        material_precision, material_recall, history_material,
        checkpoint_dir
    )
    
    # Save model
    try:
        model.save(model_save_path)
        logger.info(f"Full model saved to {model_save_path}")
    except Exception as e:
        logger.error(f"Could not save full model: {str(e)}")
        logger.info("Attempting to save sub-models...")
        try:
            model.student_model.save(student_model_save_path)
            model.course_model.save(course_model_save_path)
            model.tutor_model.save(tutor_model_save_path)
            model.material_model.save(material_model_save_path)
            logger.info(f"Sub-models saved to {checkpoint_dir}")
        except Exception as e2:
            logger.error(f"Error saving sub-models: {str(e2)}")
    
    # Save bruteforce data
    try:
        bruteforce_data_path = os.path.join(checkpoint_dir, 'bruteforce_data.npz')
        model.save_bruteforce_data(bruteforce_data_path)
        logger.info(f"BruteForce data saved to {bruteforce_data_path}")
    except Exception as e:
        logger.error(f"Could not save bruteforce data: {str(e)}")
    
    def plot_training_history(history, title, precision, recall):
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f"{title} (Precision@10: {precision:.4f}, Recall@10: {recall:.4f})")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plot_path = os.path.join(checkpoint_dir, f'{title.lower().replace(" ", "_")}.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved training history plot: {plot_path}")
    
    plot_training_history(history_course, 'Course Model Training', course_precision, course_recall)
    plot_training_history(history_tutor, 'Tutor Model Training', tutor_precision, tutor_recall)
    plot_training_history(history_material, 'Material Model Training', material_precision, material_recall)
    
    logger.info("Verifying model serialization...")
    model.build_metrics()
    try:
        temp_save_path = os.path.join(checkpoint_dir, 'temp_save')
        model.save(temp_save_path)
        logger.info("Model serialization test passed")
        shutil.rmtree(temp_save_path)  # Clean up temporary save
    except Exception as e:
        logger.error(f"Model serialization test failed: {str(e)}")

    return history_course, history_tutor, history_material

if __name__ == "__main__":
    try:
        history_course, history_tutor, history_material = train_and_evaluate()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")