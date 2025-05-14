import os
import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.recommender import RecommendationModel, load_and_preprocess_data
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        for col in ['Khối Lớp hiện tại', 'Khối Lớp']:
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
        (data['student_material_train_dataset'], 'material_train')
    ]:
        for batch in dataset.take(1):
            logger.info(f"{name} batch keys: {list(batch.keys())}")
            logger.info(f"{name} batch shapes: {{k: v.shape for k, v in batch.items()}}")

    # Prepare datasets
    course_train_dataset = data['student_course_train_dataset'].batch(32).cache().prefetch(tf.data.AUTOTUNE)
    course_test_dataset = data['student_course_test_dataset'].batch(32).cache().prefetch(tf.data.AUTOTUNE)
    tutor_train_dataset = data['student_tutor_train_dataset'].batch(32).cache().prefetch(tf.data.AUTOTUNE)
    tutor_test_dataset = data['student_tutor_test_dataset'].batch(32).cache().prefetch(tf.data.AUTOTUNE)
    material_train_dataset = data['student_material_train_dataset'].batch(32).cache().prefetch(tf.data.AUTOTUNE)
    material_test_dataset = data['student_material_test_dataset'].batch(32).cache().prefetch(tf.data.AUTOTUNE)
    
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
    
    # Setup checkpoint directory
    checkpoint_dir = os.path.normpath(os.path.join(os.getcwd(), 'checkpoints'))
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Checkpoint directory created/verified at: {checkpoint_dir}")
    except Exception as e:
        logger.error(f"Failed to create checkpoint directory: {str(e)}")
        raise
    
    # Setup model saving paths
    model_save_path = os.path.join(checkpoint_dir, 'saved_model')
    student_model_save_path = os.path.join(checkpoint_dir, 'student_model')
    course_model_save_path = os.path.join(checkpoint_dir, 'course_model')
    tutor_model_save_path = os.path.join(checkpoint_dir, 'tutor_model')
    material_model_save_path = os.path.join(checkpoint_dir, 'material_model')
    
    # Setup checkpointing
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'checkpoint_epoch_{epoch}'),
        save_weights_only=True,
        save_best_only=False,
        monitor='loss',
        verbose=1
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
    
    epochs = 20
    
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
    
    # Save model with error handling
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
    
    # Save bruteforce data with error handling
    try:
        bruteforce_data_path = os.path.join(checkpoint_dir, 'bruteforce_data.npz')
        model.save_bruteforce_data(bruteforce_data_path)
        logger.info(f"BruteForce data saved to {bruteforce_data_path}")
    except Exception as e:
        logger.error(f"Could not save bruteforce data: {str(e)}")
    
    def plot_training_history(history, title):
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plot_path = os.path.join(checkpoint_dir, f'{title.lower().replace(" ", "_")}.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved training history plot: {plot_path}")
    
    plot_training_history(history_course, 'Course Model Training')
    plot_training_history(history_tutor, 'Tutor Model Training')
    plot_training_history(history_material, 'Material Model Training')
    
    return history_course, history_tutor, history_material

if __name__ == "__main__":
    try:
        history_course, history_tutor, history_material = train_and_evaluate()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
