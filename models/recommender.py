import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np

# Print versions for debugging
print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow Recommenders version: {tfrs.__version__}")

class CustomFactorizedTopK(tfrs.metrics.FactorizedTopK):
    def __init__(self, candidate_embeddings, candidate_ids, id_lookup, ks=[1, 5, 10, 100], name="custom_factorized_top_k"):
        super().__init__(candidates=None, ks=ks, name=name)  # No candidates passed to parent
        self.candidate_embeddings = tf.convert_to_tensor(candidate_embeddings, dtype=tf.float32)
        self.candidate_ids = tf.convert_to_tensor(candidate_ids, dtype=tf.string)
        self.id_lookup = id_lookup
        self._ks = ks
        self._top_k_metrics = {
            k: tf.keras.metrics.Mean(f'top_{k}_accuracy') for k in ks
        }

    def update_state(self, query_embeddings, candidate_embeddings=None, true_candidate_ids=None):
        # Compute scores: dot product between query and candidate embeddings
        scores = tf.matmul(query_embeddings, self.candidate_embeddings, transpose_b=True)  # (batch_size, num_candidates)
        
        # Get top-k scores and indices
        top_k_predictions, retrieved_indices = tf.nn.top_k(scores, k=max(self._ks))
        
        # Convert retrieved_indices (int32/int64) to string IDs using id_lookup
        retrieved_indices = tf.cast(retrieved_indices, tf.int64)  # Ensure int64 for StringLookup
        retrieved_ids = self.id_lookup(retrieved_indices)  # Should output strings
        
        # Debug: Print shapes and types
        tf.print("Retrieved indices shape:", tf.shape(retrieved_indices))
        tf.print("Retrieved indices dtype:", retrieved_indices.dtype)
        tf.print("Sample retrieved indices:", retrieved_indices[:5])
        tf.print("Retrieved IDs shape:", tf.shape(retrieved_ids))
        tf.print("Retrieved IDs dtype:", retrieved_ids.dtype)
        tf.print("Sample retrieved IDs:", retrieved_ids[:5])
        if true_candidate_ids is not None:
            tf.print("True candidate IDs shape:", tf.shape(true_candidate_ids))
            tf.print("True candidate IDs dtype:", true_candidate_ids.dtype)
            tf.print("Sample true candidate IDs:", true_candidate_ids[:5])
        
        # Ensure true_candidate_ids and retrieved_ids are compatible
        update_ops = []
        if true_candidate_ids is not None:
            # Reshape true_candidate_ids to match retrieved_ids
            true_candidate_ids = tf.expand_dims(true_candidate_ids, -1)  # (batch_size,) -> (batch_size, 1)
            # Ensure both are strings
            true_candidate_ids = tf.cast(true_candidate_ids, tf.string)
            retrieved_ids = tf.cast(retrieved_ids, tf.string)
            # Compute matches
            ids_match = tf.cast(tf.math.equal(true_candidate_ids, retrieved_ids), tf.float32)
            tf.print("ids_match shape:", tf.shape(ids_match))
            tf.print("Sample ids_match:", ids_match[:5])
            # Update metrics
            for k in self._ks:
                metric_value = tf.reduce_mean(ids_match[:, :k], axis=1)  # (batch_size,)
                tf.print(f"Metric value top_{k} shape:", tf.shape(metric_value))
                tf.print(f"Sample metric value top_{k}:", metric_value[:5])
                update_op = self._top_k_metrics[k].update_state(metric_value)
                update_ops.append(update_op)
        else:
            # Update metrics with zeros if no true_candidate_ids
            for k in self._ks:
                metric_value = tf.zeros(tf.shape(query_embeddings)[0], dtype=tf.float32)
                update_op = self._top_k_metrics[k].update_state(metric_value)
                update_ops.append(update_op)

        return update_ops

    def result(self):
        return {f'top_{k}_accuracy': metric.result() for k, metric in self._top_k_metrics.items()}

    def reset_states(self):
        for metric in self._top_k_metrics.values():
            metric.reset_states()

class StudentTower(tf.keras.Model):
    def __init__(self, unique_schools, unique_grades, unique_goals, unique_favorite_subjects, unique_learning_methods):
        super().__init__()
        
        # Embedding layers for categorical attributes
        self.school_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_schools, mask_token=None),
            tf.keras.layers.Embedding(len(unique_schools) + 1, 32)
        ])
        
        self.grade_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_grades, mask_token=None),
            tf.keras.layers.Embedding(len(unique_grades) + 1, 32)
        ])
        
        self.goal_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_goals, mask_token=None),
            tf.keras.layers.Embedding(len(unique_goals) + 1, 32)
        ])
        
        self.favorite_subject_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_favorite_subjects, mask_token=None),
            tf.keras.layers.Embedding(len(unique_favorite_subjects) + 1, 32)
        ])
        
        self.learning_method_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_learning_methods, mask_token=None),
            tf.keras.layers.Embedding(len(unique_learning_methods) + 1, 32)
        ])
        
        # Dense layers
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(32)
        ])

    def call(self, inputs):
        school_embedding = self.school_embedding(inputs['Trường học hiện tại'])
        grade_embedding = self.grade_embedding(inputs['Khối Lớp hiện tại'])
        goal_embedding = self.goal_embedding(inputs['Mục tiêu học'])
        favorite_subject_embedding = self.favorite_subject_embedding(inputs['Môn học yêu thích'])
        learning_method_embedding = self.learning_method_embedding(inputs['Phương pháp học yêu thích'])
        
        return self.dense_layers(
            tf.concat([
                school_embedding, grade_embedding, goal_embedding,
                favorite_subject_embedding, learning_method_embedding
            ], axis=1)
        )

class CourseModel(tf.keras.Model):
    def __init__(self, unique_centers, unique_subjects, unique_grades, unique_methods):
        super().__init__()
        
        # Embedding layers
        self.center_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_centers, mask_token=None),
            tf.keras.layers.Embedding(len(unique_centers) + 1, 32)
        ])
        
        self.subject_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_subjects, mask_token=None),
            tf.keras.layers.Embedding(len(unique_subjects) + 1, 32)
        ])
        
        self.grade_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_grades, mask_token=None),
            tf.keras.layers.Embedding(len(unique_grades) + 1, 32)
        ])
        
        self.method_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_methods, mask_token=None),
            tf.keras.layers.Embedding(len(unique_methods) + 1, 32)
        ])
        
        # Dense layers for numeric attributes
        self.cost_dense = tf.keras.layers.Dense(32)
        self.time_dense = tf.keras.layers.Dense(32)
        
        # Dense layers
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(32)
        ])
        
        # Candidate embeddings
        self.candidate_embeddings = None

    def call(self, inputs):
        # Cast numeric inputs to float32
        cost = tf.cast(inputs['Chi phí'], tf.float32)
        time = tf.cast(inputs['Thời gian'], tf.float32)
        
        # Ensure 2D input for dense layers (batch_size, 1)
        cost_2d = tf.expand_dims(cost, axis=-1)  # (batch_size,) -> (batch_size, 1) or () -> (1,)
        time_2d = tf.expand_dims(time, axis=-1)
        
        # If input is scalar, add batch dimension to make (1, 1)
        if len(cost_2d.shape) == 1:
            cost_2d = tf.expand_dims(cost_2d, axis=0)  # (1,) -> (1, 1)
        if len(time_2d.shape) == 1:
            time_2d = tf.expand_dims(time_2d, axis=0)  # (1,) -> (1, 1)
        
        # Get embeddings (batch_size, 32) or (32,) for single sample
        center_embedding = self.center_embedding(inputs['Tên Trung Tâm'])
        subject_embedding = self.subject_embedding(inputs['Môn học'])
        grade_embedding = self.grade_embedding(inputs['Khối Lớp'])
        method_embedding = self.method_embedding(inputs['Phương pháp học'])
        
        # Ensure all embeddings are rank 2 (batch_size, 32)
        if len(center_embedding.shape) == 1:
            center_embedding = tf.expand_dims(center_embedding, axis=0)  # (32,) -> (1, 32)
        if len(subject_embedding.shape) == 1:
            subject_embedding = tf.expand_dims(subject_embedding, axis=0)
        if len(grade_embedding.shape) == 1:
            grade_embedding = tf.expand_dims(grade_embedding, axis=0)
        if len(method_embedding.shape) == 1:
            method_embedding = tf.expand_dims(method_embedding, axis=0)
        
        # Process numeric inputs through dense layers (batch_size, 32) or (1, 32)
        cost_embedding = self.cost_dense(cost_2d)
        time_embedding = self.time_dense(time_2d)
        
        # Ensure embeddings are rank 2
        center_embedding = tf.ensure_shape(center_embedding, [None, 32])
        subject_embedding = tf.ensure_shape(subject_embedding, [None, 32])
        grade_embedding = tf.ensure_shape(grade_embedding, [None, 32])
        method_embedding = tf.ensure_shape(method_embedding, [None, 32])
        cost_embedding = tf.ensure_shape(cost_embedding, [None, 32])
        time_embedding = tf.ensure_shape(time_embedding, [None, 32])
        
        # Concatenate all embeddings along the feature dimension
        concatenated = tf.concat([
            center_embedding, subject_embedding, grade_embedding,
            method_embedding, cost_embedding, time_embedding
        ], axis=1)
        
        return self.dense_layers(concatenated)

class TutorModel(tf.keras.Model):
    def __init__(self, unique_tutors, unique_subjects, unique_grades, unique_teaching_times):
        super().__init__()
        
        # Embedding layers
        self.tutor_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_tutors, mask_token=None),
            tf.keras.layers.Embedding(len(unique_tutors) + 1, 32)
        ])
        
        self.subject_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_subjects, mask_token=None),
            tf.keras.layers.Embedding(len(unique_subjects) + 1, 32)
        ])
        
        self.grade_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_grades, mask_token=None),
            tf.keras.layers.Embedding(len(unique_grades) + 1, 32)
        ])
        
        self.teaching_time_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_teaching_times, mask_token=None),
            tf.keras.layers.Embedding(len(unique_teaching_times) + 1, 32)
        ])
        
        # Dense layers for experience
        self.experience_dense = tf.keras.layers.Dense(32)
        
        # Dense layers
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(32)
        ])
        
        # Candidate embeddings
        self.candidate_embeddings = None

    def call(self, inputs):
        # Debug: Print input keys
        tf.print("TutorModel inputs keys:", tf.convert_to_tensor(list(inputs.keys())))
        
        # Cast numeric inputs to float32
        experience = tf.cast(inputs['Kinh nghiệm giảng dạy'], tf.float32)
        
        # Ensure 2D input for dense layers (batch_size, 1)
        experience_2d = tf.expand_dims(experience, axis=-1)  # (batch_size,) -> (batch_size, 1) or () -> (1,)
        
        # If input is scalar, add batch dimension to make (1, 1)
        if len(experience_2d.shape) == 1:
            experience_2d = tf.expand_dims(experience_2d, axis=0)  # (1,) -> (1, 1)
        
        # Get embeddings (batch_size, 32) or (32,) for single sample
        tutor_embedding = self.tutor_embedding(inputs['Tên gia sư'])
        subject_embedding = self.subject_embedding(inputs['Môn học'])
        grade_embedding = self.grade_embedding(inputs['Khối Lớp'])
        teaching_time_embedding = self.teaching_time_embedding(inputs['Thời gian dạy học'])
        
        # Ensure all embeddings are rank 2 (batch_size, 32)
        if len(tutor_embedding.shape) == 1:
            tutor_embedding = tf.expand_dims(tutor_embedding, axis=0)  # (32,) -> (1, 32)
        if len(subject_embedding.shape) == 1:
            subject_embedding = tf.expand_dims(subject_embedding, axis=0)
        if len(grade_embedding.shape) == 1:
            grade_embedding = tf.expand_dims(grade_embedding, axis=0)
        if len(teaching_time_embedding.shape) == 1:
            teaching_time_embedding = tf.expand_dims(teaching_time_embedding, axis=0)
        
        # Process numeric inputs through dense layers (batch_size, 32) or (1, 32)
        experience_embedding = self.experience_dense(experience_2d)
        
        # Ensure embeddings are rank 2
        tutor_embedding = tf.ensure_shape(tutor_embedding, [None, 32])
        subject_embedding = tf.ensure_shape(subject_embedding, [None, 32])
        grade_embedding = tf.ensure_shape(grade_embedding, [None, 32])
        teaching_time_embedding = tf.ensure_shape(teaching_time_embedding, [None, 32])
        experience_embedding = tf.ensure_shape(experience_embedding, [None, 32])
        
        # Concatenate all embeddings along the feature dimension
        concatenated = tf.concat([
            tutor_embedding, subject_embedding, grade_embedding,
            teaching_time_embedding, experience_embedding
        ], axis=1)
        
        return self.dense_layers(concatenated)

class MaterialModel(tf.keras.Model):
    def __init__(self, unique_materials, unique_subjects, unique_grades, unique_types):
        super().__init__()
        
        # Embedding layers
        self.material_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_materials, mask_token=None),
            tf.keras.layers.Embedding(len(unique_materials) + 1, 32)
        ])
        
        self.subject_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_subjects, mask_token=None),
            tf.keras.layers.Embedding(len(unique_subjects) + 1, 32)
        ])
        
        self.grade_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_grades, mask_token=None),
            tf.keras.layers.Embedding(len(unique_grades) + 1, 32)
        ])
        
        self.type_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_types, mask_token=None),
            tf.keras.layers.Embedding(len(unique_types) + 1, 32)
        ])
        
        # Dense layers
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(32)
        ])
        
        # Candidate embeddings
        self.candidate_embeddings = None

    def call(self, inputs):
        # Get embeddings (batch_size, 32) or (32,) for single sample
        material_embedding = self.material_embedding(inputs['Tên tài liệu'])
        subject_embedding = self.subject_embedding(inputs['Môn học'])
        grade_embedding = self.grade_embedding(inputs['Khối Lớp'])
        type_embedding = self.type_embedding(inputs['Loại tài liệu'])
        
        # Ensure all embeddings are rank 2 (batch_size, 32)
        if len(material_embedding.shape) == 1:
            material_embedding = tf.expand_dims(material_embedding, axis=0)  # (32,) -> (1, 32)
        if len(subject_embedding.shape) == 1:
            subject_embedding = tf.expand_dims(subject_embedding, axis=0)
        if len(grade_embedding.shape) == 1:
            grade_embedding = tf.expand_dims(grade_embedding, axis=0)
        if len(type_embedding.shape) == 1:
            type_embedding = tf.expand_dims(type_embedding, axis=0)
        
        # Ensure embeddings are rank 2
        material_embedding = tf.ensure_shape(material_embedding, [None, 32])
        subject_embedding = tf.ensure_shape(subject_embedding, [None, 32])
        grade_embedding = tf.ensure_shape(grade_embedding, [None, 32])
        type_embedding = tf.ensure_shape(type_embedding, [None, 32])
        
        # Concatenate all embeddings along the feature dimension
        concatenated = tf.concat([
            material_embedding, subject_embedding,
            grade_embedding, type_embedding
        ], axis=1)
        
        return self.dense_layers(concatenated)

class RecommendationModel(tfrs.Model):
    def __init__(self, student_features, course_features, tutor_features, material_features,
                 student_course_train, student_tutor_train, student_material_train):
        super().__init__()
        
        # Get unique values for embeddings and convert to list of strings
        unique_schools = [str(x) for x in student_features['Trường học hiện tại'].unique()]
        unique_grades = [str(x) for x in student_features['Khối Lớp hiện tại'].unique()]
        unique_goals = [str(x) for x in student_features['Mục tiêu học'].unique()]
        unique_favorite_subjects = [str(x) for x in student_features['Môn học yêu thích'].unique()]
        unique_learning_methods = [str(x) for x in student_features['Phương pháp học yêu thích'].unique()]
        
        unique_centers = [str(x) for x in course_features['Tên Trung Tâm'].unique()]
        unique_course_subjects = [str(x) for x in course_features['Môn học'].unique()]
        unique_course_grades = [str(x) for x in course_features['Khối Lớp'].unique()]
        unique_course_methods = [str(x) for x in course_features['Phương pháp học'].unique()]
        
        unique_tutors = [str(x) for x in tutor_features['Tên gia sư'].unique()]
        unique_tutor_subjects = [str(x) for x in tutor_features['Môn học'].unique()]
        unique_tutor_grades = [str(x) for x in tutor_features['Khối Lớp'].unique()]
        unique_teaching_times = [str(x) for x in tutor_features['Thời gian dạy học'].unique()]
        
        unique_materials = [str(x) for x in material_features['Tên tài liệu'].unique()]
        unique_material_subjects = [str(x) for x in material_features['Môn học'].unique()]
        unique_material_grades = [str(x) for x in material_features['Khối Lớp'].unique()]
        unique_material_types = [str(x) for x in material_features['Loại tài liệu'].unique()]
        
        # Initialize towers
        self.student_model = StudentTower(
            unique_schools, unique_grades, unique_goals,
            unique_favorite_subjects, unique_learning_methods
        )
        
        self.course_model = CourseModel(
            unique_centers, unique_course_subjects,
            unique_course_grades, unique_course_methods
        )
        
        self.tutor_model = TutorModel(
            unique_tutors, unique_tutor_subjects,
            unique_tutor_grades, unique_teaching_times
        )
        
        self.material_model = MaterialModel(
            unique_materials, unique_material_subjects,
            unique_material_grades, unique_material_types
        )
        
        # Create candidate embeddings with identifiers
        course_dataset = tf.data.Dataset.from_tensor_slices(dict(course_features)).batch(32)
        course_embeddings = []
        course_ids = []
        for batch in course_dataset:
            course_embeddings.append(self.course_model(batch))
            course_ids.append(tf.strings.join(['course_', tf.strings.as_string(batch['ID Trung Tâm'])]))
        self.course_model.candidate_embeddings = {
            'embeddings': tf.concat(course_embeddings, axis=0),
            'identifiers': tf.concat(course_ids, axis=0)
        }
        
        tutor_dataset = tf.data.Dataset.from_tensor_slices({
            k: v for k, v in dict(tutor_features).items()
            if k in ['ID Gia Sư', 'Tên gia sư', 'Môn học', 'Khối Lớp', 'Thời gian dạy học', 'Kinh nghiệm giảng dạy']
        }).batch(32)
        tutor_embeddings = []
        tutor_ids = []
        for batch in tutor_dataset:
            tutor_embeddings.append(self.tutor_model(batch))
            tutor_ids.append(tf.strings.join(['tutor_', tf.strings.as_string(batch['ID Gia Sư'])]))
        self.tutor_model.candidate_embeddings = {
            'embeddings': tf.concat(tutor_embeddings, axis=0),
            'identifiers': tf.concat(tutor_ids, axis=0)
        }
        
        material_dataset = tf.data.Dataset.from_tensor_slices(dict(material_features)).batch(32)
        material_embeddings = []
        material_ids = []
        for batch in material_dataset:
            material_embeddings.append(self.material_model(batch))
            material_ids.append(tf.strings.join(['material_', tf.strings.as_string(batch['ID Tài Liệu'])]))
        self.material_model.candidate_embeddings = {
            'embeddings': tf.concat(material_embeddings, axis=0),
            'identifiers': tf.concat(material_ids, axis=0)
        }
        
        # Combine all candidate embeddings and identifiers
        self.all_candidate_embeddings = tf.concat([
            self.course_model.candidate_embeddings['embeddings'],
            self.tutor_model.candidate_embeddings['embeddings'],
            self.material_model.candidate_embeddings['embeddings']
        ], axis=0)
        all_candidate_ids = tf.concat([
            self.course_model.candidate_embeddings['identifiers'],
            self.tutor_model.candidate_embeddings['identifiers'],
            self.material_model.candidate_embeddings['identifiers']
        ], axis=0)
        
        # Debug: Check for duplicate IDs
        unique_ids, counts = np.unique(all_candidate_ids.numpy(), return_counts=True)
        duplicates = unique_ids[counts > 1]
        if len(duplicates) > 0:
            print(f"Warning: Duplicate IDs found: {duplicates}")
        else:
            print("All candidate IDs are unique.")
        
        # Debug: Print shapes and sample IDs
        print(f"All candidate embeddings shape: {self.all_candidate_embeddings.shape}")
        print(f"All candidate IDs shape: {all_candidate_ids.shape}")
        print(f"Sample candidate IDs: {all_candidate_ids[:5]}")
        print(f"All candidate IDs dtype: {all_candidate_ids.dtype}")
        
        # Create a lookup table to map integer indices to string IDs
        self.id_lookup = tf.keras.layers.StringLookup(
            vocabulary=all_candidate_ids.numpy(),
            mask_token=None,
            num_oov_indices=0,
            output_mode='int'  # Output indices
        )
        # Create a reverse lookup to map indices back to strings
        self.reverse_id_lookup = tf.keras.layers.StringLookup(
            vocabulary=all_candidate_ids.numpy(),
            mask_token=None,
            num_oov_indices=0,
            invert=True  # Maps indices to strings
        )
        
        # Use CustomFactorizedTopK with precomputed embeddings and IDs
        self.task = tfrs.tasks.Retrieval(
            metrics=CustomFactorizedTopK(
                candidate_embeddings=self.all_candidate_embeddings,
                candidate_ids=all_candidate_ids,
                id_lookup=self.reverse_id_lookup
            )
        )

        # Create training datasets
        self.course_train_dataset = tf.data.Dataset.from_tensor_slices(dict(student_course_train))
        self.tutor_train_dataset = tf.data.Dataset.from_tensor_slices(dict(student_tutor_train))
        self.material_train_dataset = tf.data.Dataset.from_tensor_slices(dict(student_material_train))

    def call(self, inputs):
        # Get student embeddings
        student_embeddings = self.student_model(inputs)
        
        # Get candidate embeddings based on input type
        if 'ID Trung Tâm' in inputs:
            candidate_embeddings = self.course_model(inputs)
        elif 'ID Gia Sư' in inputs:
            candidate_embeddings = self.tutor_model(inputs)
        elif 'ID Tài Liệu' in inputs:
            candidate_embeddings = self.material_model(inputs)
        else:
            raise ValueError("Input must contain either ID Trung Tâm, ID Gia Sư, or ID Tài Liệu")
        
        return student_embeddings, candidate_embeddings

    def compute_loss(self, features, training=False):
        print("Starting compute_loss...")
        # Debug: Print input keys
        print("compute_loss input keys:", list(features.keys()))
        
        # Get student embeddings
        student_embeddings = self.student_model(features)
        print(f"Student embeddings shape: {student_embeddings.shape}")
        
        # Get candidate embeddings and IDs based on input type
        if 'ID Trung Tâm' in features:
            candidate_embeddings = self.course_model(features)
            candidate_ids = tf.strings.join(['course_', tf.strings.as_string(features['ID Trung Tâm'])])
        elif 'ID Gia Sư' in features:
            candidate_embeddings = self.tutor_model(features)
            candidate_ids = tf.strings.join(['tutor_', tf.strings.as_string(features['ID Gia Sư'])])
        elif 'ID Tài Liệu' in features:
            candidate_embeddings = self.material_model(features)
            candidate_ids = tf.strings.join(['material_', tf.strings.as_string(features['ID Tài Liệu'])])
        else:
            raise ValueError("Input must contain either ID Trung Tâm, ID Gia Sư, or ID Tài Liệu")
        
        print(f"Candidate embeddings shape: {candidate_embeddings.shape}")
        print(f"Candidate IDs shape: {candidate_ids.shape}")
        print(f"Candidate IDs dtype: {candidate_ids.dtype}")
        print(f"Sample candidate IDs: {candidate_ids[:5]}")
        
        # Debug: Verify id_lookup output
        if training:
            sample_indices = tf.range(5, dtype=tf.int64)
            retrieved_ids = self.reverse_id_lookup(sample_indices)
            print(f"Sample indices: {sample_indices}")
            print(f"Retrieved IDs from reverse_id_lookup: {retrieved_ids}")
        
        # Compute loss using the retrieval task
        loss = self.task(
            query_embeddings=student_embeddings,
            candidate_embeddings=candidate_embeddings,
            candidate_ids=candidate_ids,
            compute_metrics=training
        )
        print(f"Loss computed: {loss}")
        
        # Debug metrics
        if training:
            for metric in self.task.metrics:
                print(f"Metric {metric.name}: {metric.result()}")
        
        return loss

def load_and_preprocess_data():
    # Define categorical columns to be read as strings
    categorical_dtypes = {col: str for col in [
        'Trường học hiện tại', 'Khối Lớp hiện tại', 'Mục tiêu học',
        'Môn học yêu thích', 'Phương pháp học yêu thích', 'Tên Trung Tâm',
        'Môn học', 'Khối Lớp', 'Phương pháp học', 'Tên gia sư',
        'Thời gian dạy học', 'Tên tài liệu', 'Loại tài liệu'
    ]}
    
    # Load data with specified dtypes
    student_data = pd.read_csv('data/hoc_sinh.csv', dtype=categorical_dtypes)
    course_data = pd.read_csv('data/trung_tam.csv', dtype=categorical_dtypes)
    tutor_data = pd.read_csv('data/gia_su.csv', dtype=categorical_dtypes)
    material_data = pd.read_csv('data/tai_lieu.csv', dtype=categorical_dtypes)
    
    # Drop 'Phương pháp dạy' from tutor_data if present
    if 'Phương pháp dạy' in tutor_data.columns:
        tutor_data = tutor_data.drop(columns=['Phương pháp dạy'])
        print("Dropped 'Phương pháp dạy' from tutor_data")
    
    # Load interaction data
    student_course_train = pd.read_csv('data/hoc_sinh_trung_tam_train.csv', dtype=categorical_dtypes)
    student_course_test = pd.read_csv('data/hoc_sinh_trung_tam_test.csv', dtype=categorical_dtypes)
    student_tutor_train = pd.read_csv('data/hoc_sinh_gia_su_train.csv', dtype=categorical_dtypes)
    student_tutor_test = pd.read_csv('data/hoc_sinh_gia_su_test.csv', dtype=categorical_dtypes)
    student_material_train = pd.read_csv('data/hoc_sinh_tai_lieu_train.csv', dtype=categorical_dtypes)
    student_material_test = pd.read_csv('data/hoc_sinh_tai_lieu_test.csv', dtype=categorical_dtypes)
    
    # Convert numeric columns to float
    numeric_columns = {
        'Chi phí': 'float32',
        'Thời gian': 'float32',
        'Kinh nghiệm giảng dạy': 'float32'
    }
    
    # Convert categorical columns to string
    categorical_columns = [
        'Trường học hiện tại', 'Khối Lớp hiện tại', 'Mục tiêu học',
        'Môn học yêu thích', 'Phương pháp học yêu thích', 'Tên Trung Tâm',
        'Môn học', 'Khối Lớp', 'Phương pháp học', 'Tên gia sư',
        'Thời gian dạy học', 'Tên tài liệu', 'Loại tài liệu'
    ]
    
    def preprocess_categorical(df):
        # Convert categorical columns to string and ensure they are not numeric
        for col in categorical_columns:
            if col in df.columns:
                # Convert to string first
                df[col] = df[col].astype(str)
                # Add prefix to ensure string format
                df[col] = 'cat_' + df[col].str.strip()
                # Replace any remaining numeric values with string format
                df[col] = df[col].apply(lambda x: f'cat_{x}' if x.replace('cat_', '').isdigit() else x)
                # Ensure all values are strings and not numeric
                df[col] = df[col].apply(lambda x: str(x) if not isinstance(x, str) else x)
                print(f"Column {col} dtype after preprocessing: {df[col].dtype}")
        return df
    
    def preprocess_numeric(df):
        # Convert numeric columns
        for col, dtype in numeric_columns.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(dtype)
        return df
    
    # Preprocess all dataframes
    dataframes = [
        student_data, course_data, tutor_data, material_data,
        student_course_train, student_course_test,
        student_tutor_train, student_tutor_test,
        student_material_train, student_material_test
    ]
    
    for df in dataframes:
        df = preprocess_categorical(df)
        df = preprocess_numeric(df)
    
    # Print sample values for debugging
    print("\nSample values after preprocessing:")
    for df_name, df in [
        ('student_data', student_data),
        ('course_data', course_data),
        ('tutor_data', tutor_data),
        ('material_data', material_data)
    ]:
        print(f"\n{df_name} sample values:")
        for col in categorical_columns:
            if col in df.columns:
                print(f"{col}: {df[col].iloc[0]}")
        for col in numeric_columns:
            if col in df.columns:
                print(f"{col}: {df[col].iloc[0]}")
    
    # Convert dataframes to TensorFlow datasets
    def create_tf_dataset(df):
        # Ensure all categorical columns are strings
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        return tf.data.Dataset.from_tensor_slices(dict(df))
    
    # Create TensorFlow datasets
    student_course_train_dataset = create_tf_dataset(student_course_train)
    student_tutor_train_dataset = create_tf_dataset(student_tutor_train)
    student_material_train_dataset = create_tf_dataset(student_material_train)
    
    return {
        'student_data': student_data,
        'course_data': course_data,
        'tutor_data': tutor_data,
        'material_data': material_data,
        'student_course_train': student_course_train,
        'student_course_test': student_course_test,
        'student_tutor_train': student_tutor_train,
        'student_tutor_test': student_tutor_test,
        'student_material_train': student_material_train,
        'student_material_test': student_material_test,
        'student_course_train_dataset': student_course_train_dataset,
        'student_tutor_train_dataset': student_tutor_train_dataset,
        'student_material_train_dataset': student_material_train_dataset
    }
