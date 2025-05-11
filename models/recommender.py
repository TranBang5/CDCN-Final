import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np

print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow Recommenders version: {tfrs.__version__}")

def load_and_preprocess_data():
    categorical_dtypes = {col: str for col in [
        'Trường học hiện tại', 'Khối Lớp hiện tại', 'Mục tiêu học',
        'Môn học yêu thích', 'Phương pháp học yêu thích', 'Tên Trung Tâm',
        'Môn học', 'Khối Lớp', 'Phương pháp học', 'Tên gia sư',
        'Thời gian dạy học', 'Tên tài liệu', 'Loại tài liệu', 'Địa chỉ'
    ]}
    
    student_data = pd.read_csv('data/hoc_sinh.csv', dtype=categorical_dtypes, encoding='utf-8')
    course_data = pd.read_csv('data/trung_tam.csv', dtype=categorical_dtypes, encoding='utf-8')
    tutor_data = pd.read_csv('data/gia_su.csv', dtype=categorical_dtypes, encoding='utf-8')
    material_data = pd.read_csv('data/tai_lieu.csv', dtype=categorical_dtypes, encoding='utf-8')
    
    if 'Phương pháp dạy' in tutor_data.columns:
        tutor_data = tutor_data.drop(columns=['Phương pháp dạy'])
        print("Dropped 'Phương pháp dạy' from tutor_data")
    
    student_course_train = pd.read_csv('data/hoc_sinh_trung_tam_train.csv', dtype=categorical_dtypes, encoding='utf-8')
    student_course_test = pd.read_csv('data/hoc_sinh_trung_tam_test.csv', dtype=categorical_dtypes, encoding='utf-8')
    student_tutor_train = pd.read_csv('data/hoc_sinh_gia_su_train.csv', dtype=categorical_dtypes, encoding='utf-8')
    student_tutor_test = pd.read_csv('data/hoc_sinh_gia_su_test.csv', dtype=categorical_dtypes, encoding='utf-8')
    student_material_train = pd.read_csv('data/hoc_sinh_tai_lieu_train.csv', dtype=categorical_dtypes, encoding='utf-8')
    student_material_test = pd.read_csv('data/hoc_sinh_tai_lieu_test.csv', dtype=categorical_dtypes, encoding='utf-8')
    
    id_columns = ['ID Học Sinh', 'ID Trung Tâm', 'ID Gia Sư', 'ID Tài Liệu']
    for df in [student_data, course_data, tutor_data, material_data,
               student_course_train, student_course_test,
               student_tutor_train, student_tutor_test,
               student_material_train, student_material_test]:
        for col in id_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                df[col] = df[col].astype(str)
    
    print("\nDataset Statistics:")
    print("Student dataset shape:", student_data.shape)
    print("Course dataset shape:", course_data.shape)
    print("Tutor dataset shape:", tutor_data.shape)
    print("Material dataset shape:", material_data.shape)
    print("Student-Course train interactions:", student_course_train.shape)
    print("Student-Tutor train interactions:", student_tutor_train.shape)
    print("Student-Material train interactions:", student_material_train.shape)
    print("Student-Course test interactions:", student_course_test.shape)
    print("Student-Tutor test interactions:", student_tutor_test.shape)
    print("Student-Material test interactions:", student_material_test.shape)
    
    print("\nMaterial ID distribution:")
    print(student_material_train['ID Tài Liệu'].value_counts().head())
    print("\nMissing values in material dataset:")
    print(material_data.isnull().sum())
    print("\nMissing values in student-material interactions:")
    print(student_material_train.isnull().sum())
    
    column_mappings = [
        ('Khối Lớp hiện tại', 'Khối Lớp'),
        ('Môn học yêu thích', 'Môn học')
    ]
    
    for student_col, other_col in column_mappings:
        if student_col in student_data.columns and other_col in material_data.columns:
            student_vocab = set(student_data[student_col].astype(str).unique())
            material_vocab = set(material_data[other_col].astype(str).unique())
            print(f"\nVocabulary overlap ({student_col} vs {other_col}):")
            print(f"Overlap size:", len(student_vocab & material_vocab))
            print(f"Unique in {student_col} (student):", len(student_vocab))
            print(f"Unique in {other_col} (material):", len(material_vocab))
    
    numeric_columns = {
        'Chi phí': 'float32',
        'Thời gian': 'float32',
        'Kinh nghiệm giảng dạy': 'float32',
        'Đánh giá': 'float64'
    }
    
    categorical_columns = [
        'Trường học hiện tại', 'Khối Lớp hiện tại', 'Mục tiêu học',
        'Môn học yêu thích', 'Phương pháp học yêu thích', 'Tên Trung Tâm',
        'Môn học', 'Khối Lớp', 'Phương pháp học', 'Tên gia sư',
        'Thời gian dạy học', 'Tên tài liệu', 'Loại tài liệu', 'Địa chỉ'
    ]
    
    def preprocess_categorical(df):
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
                df[col] = df[col].str.encode('utf-8').str.decode('utf-8')
                df[col] = 'cat_' + df[col]
                df[col] = df[col].apply(lambda x: f'cat_{x}' if x.replace('cat_', '').isdigit() else x)
                print(f"Column {col} dtype after preprocessing: {df[col].dtype}")
        return df
    
    def preprocess_numeric(df):
        for col, dtype in numeric_columns.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(dtype)
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
        return df
    
    dataframes = [
        student_data, course_data, tutor_data, material_data,
        student_course_train, student_course_test,
        student_tutor_train, student_tutor_test,
        student_material_train, student_material_test
    ]
    
    for df in dataframes:
        df = preprocess_categorical(df)
        df = preprocess_numeric(df)
    
    print("\nSample values after preprocessing:")
    for df_name, df in [
        ('student_data', student_data),
        ('course_data', course_data),
        ('tutor_data', tutor_data),
        ('material_data', material_data),
        ('student_course_train', student_course_train)
    ]:
        print(f"\n{df_name} sample values:")
        for col in categorical_columns:
            if col in df.columns:
                print(f"{col}: {df[col].iloc[0]}")
        for col in numeric_columns:
            if col in df.columns:
                print(f"{col}: {df[col].iloc[0]}")
        for col in id_columns:
            if col in df.columns:
                print(f"{col}: {df[col].iloc[0]}")
    
    def create_tf_dataset(df):
        dtype_dict = {}
        for col in df.columns:
            if col in categorical_columns or col in id_columns or col in ['Tên']:
                dtype_dict[col] = tf.string
            elif col in numeric_columns:
                dtype_dict[col] = tf.float32 if numeric_columns[col] == 'float32' else tf.float64
            else:
                dtype_dict[col] = tf.string
        
        data_dict = {col: df[col].values for col in df.columns}
        for col, dtype in dtype_dict.items():
            if dtype == tf.string:
                data_dict[col] = np.array([x.encode('utf-8') if isinstance(x, str) and x else b'unknown' for x in data_dict[col]], dtype=np.bytes_)
            elif dtype == tf.float32:
                data_dict[col] = np.array(data_dict[col], dtype=np.float32)
            elif dtype == tf.float64:
                data_dict[col] = np.array(data_dict[col], dtype=np.float64)
        
        dataset = tf.data.Dataset.from_tensor_slices(data_dict)
        
        for element in dataset.take(1):
            print(f"\nDataset dtypes for {df.name if hasattr(df, 'name') else 'dataset'}:")
            for key, value in element.items():
                print(f"{key}: {value.dtype}")
            print(f"\nSample values for {df.name if hasattr(df, 'name') else 'dataset'}:")
            for key, value in element.items():
                if value.dtype == tf.string:
                    print(f"{key}: {value.numpy().decode('utf-8') if value.numpy() else 'None'}")
                else:
                    print(f"{key}: {value.numpy()}")
        
        return dataset
    
    student_course_train.name = 'student_course_train'
    student_tutor_train.name = 'student_tutor_train'
    student_material_train.name = 'student_material_train'
    
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

class StudentTower(tf.keras.Model):
    def __init__(self, unique_schools, unique_grades, unique_goals, unique_favorite_subjects, unique_learning_methods):
        super().__init__()
        
        # Convert vocabularies to strings
        unique_schools = [str(x) for x in unique_schools]
        unique_grades = [str(x) for x in unique_grades]
        unique_goals = [str(x) for x in unique_goals]
        unique_favorite_subjects = [str(x) for x in unique_favorite_subjects]
        unique_learning_methods = [str(x) for x in unique_learning_methods]
        
        print(f"Unique schools: {len(unique_schools)}, Sample: {unique_schools[:5]}")
        print(f"Unique grades: {len(unique_grades)}, Sample: {unique_grades[:5]}")
        print(f"Unique goals: {len(unique_goals)}, Sample: {unique_goals[:5]}")
        print(f"Unique favorite subjects: {len(unique_favorite_subjects)}, Sample: {unique_favorite_subjects[:5]}")
        print(f"Unique learning methods: {len(unique_learning_methods)}, Sample: {unique_learning_methods[:5]}")
        
        self.school_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_schools, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.school_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_schools) + 2, output_dim=32
        )
        
        self.grade_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_grades, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.grade_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_grades) + 2, output_dim=32
        )
        
        self.goal_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_goals, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.goal_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_goals) + 2, output_dim=32
        )
        
        self.favorite_subject_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_favorite_subjects, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.favorite_subject_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_favorite_subjects) + 2, output_dim=32
        )
        
        self.learning_method_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_learning_methods, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.learning_method_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_learning_methods) + 2, output_dim=32
        )
        
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(160,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32)
        ])

    def call(self, inputs):
        # Debug inputs
        print("StudentTower input dtypes:", {k: v.dtype for k, v in inputs.items()})
        tf.print("School input sample:", inputs['Trường học hiện tại'][:5])
        tf.print("Grade input sample:", inputs['Khối Lớp hiện tại'][:5])
        tf.print("Goal input sample:", inputs['Mục tiêu học'][:5])
        tf.print("Favorite subject input sample:", inputs['Môn học yêu thích'][:5])
        tf.print("Learning method input sample:", inputs['Phương pháp học yêu thích'][:5])
        
        # Convert strings to indices
        school_indices = self.school_lookup(inputs['Trường học hiện tại'])
        grade_indices = self.grade_lookup(inputs['Khối Lớp hiện tại'])
        goal_indices = self.goal_lookup(inputs['Mục tiêu học'])
        favorite_subject_indices = self.favorite_subject_lookup(inputs['Môn học yêu thích'])
        learning_method_indices = self.learning_method_lookup(inputs['Phương pháp học yêu thích'])
        
        # Debug indices
        tf.print("School indices sample:", school_indices[:5])
        tf.print("Grade indices sample:", grade_indices[:5])
        tf.print("Goal indices sample:", goal_indices[:5])
        tf.print("Favorite subject indices sample:", favorite_subject_indices[:5])
        tf.print("Learning method indices sample:", learning_method_indices[:5])
        
        # Get embeddings
        school_embedding = self.school_embedding(school_indices)
        grade_embedding = self.grade_embedding(grade_indices)
        goal_embedding = self.goal_embedding(goal_indices)
        favorite_subject_embedding = self.favorite_subject_embedding(favorite_subject_indices)
        learning_method_embedding = self.learning_method_embedding(learning_method_indices)
        
        # Debug embedding shapes
        print("School embedding shape:", school_embedding.shape)
        print("Grade embedding shape:", grade_embedding.shape)
        print("Goal embedding shape:", goal_embedding.shape)
        print("Favorite subject embedding shape:", favorite_subject_embedding.shape)
        print("Learning method embedding shape:", learning_method_embedding.shape)
        
        # Concatenate embeddings
        concatenated = tf.concat([
            school_embedding, grade_embedding, goal_embedding,
            favorite_subject_embedding, learning_method_embedding
        ], axis=-1)
        print("Concatenated shape:", concatenated.shape)
        
        # Process through dense layers
        output = self.dense_layers(concatenated)
        print("Output shape:", output.shape)
        
        return output

class CourseModel(tf.keras.Model):
    def __init__(self, unique_centers, unique_subjects, unique_grades, unique_methods):
        super().__init__()
        
        unique_centers = [str(x) for x in unique_centers]
        unique_subjects = [str(x) for x in unique_subjects]
        unique_grades = [str(x) for x in unique_grades]
        unique_methods = [str(x) for x in unique_methods]
        
        self.center_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_centers, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.center_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_centers) + 2, output_dim=32
        )
        
        self.subject_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_subjects, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.subject_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_subjects) + 2, output_dim=32
        )
        
        self.grade_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_grades, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.grade_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_grades) + 2, output_dim=32
        )
        
        self.method_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_methods, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.method_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_methods) + 2, output_dim=32
        )
        
        self.cost_dense = tf.keras.layers.Dense(32)
        self.time_dense = tf.keras.layers.Dense(32)
        
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(192,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32)
        ])
        
        self.candidate_embeddings = None

    def call(self, inputs):
        # Debug inputs
        print("CourseModel input dtypes:", {k: v.dtype for k, v in inputs.items()})
        
        # Ensure inputs are properly typed
        cost = tf.cast(inputs['Chi phí'], tf.float32)
        time = tf.cast(inputs['Thời gian'], tf.float32)
        
        cost_2d = tf.expand_dims(cost, axis=-1)
        time_2d = tf.expand_dims(time, axis=-1)
        
        # Convert strings to indices
        center_indices = self.center_lookup(inputs['Tên Trung Tâm'])
        subject_indices = self.subject_lookup(inputs['Môn học'])
        grade_indices = self.grade_lookup(inputs['Khối Lớp'])
        method_indices = self.method_lookup(inputs['Phương pháp học'])
        
        # Get embeddings
        center_embedding = self.center_embedding(center_indices)
        subject_embedding = self.subject_embedding(subject_indices)
        grade_embedding = self.grade_embedding(grade_indices)
        method_embedding = self.method_embedding(method_indices)
        cost_embedding = self.cost_dense(cost_2d)
        time_embedding = self.time_dense(time_2d)
        
        # Debug embedding shapes
        print("Center embedding shape:", center_embedding.shape)
        print("Subject embedding shape:", subject_embedding.shape)
        print("Grade embedding shape:", grade_embedding.shape)
        print("Method embedding shape:", method_embedding.shape)
        print("Cost embedding shape:", cost_embedding.shape)
        print("Time embedding shape:", time_embedding.shape)
        
        # Concatenate embeddings
        concatenated = tf.concat([
            center_embedding, subject_embedding, grade_embedding,
            method_embedding, cost_embedding, time_embedding
        ], axis=-1)
        print("Concatenated shape:", concatenated.shape)
        
        # Process through dense layers
        output = self.dense_layers(concatenated)
        print("Output shape:", output.shape)
        
        return output

class TutorModel(tf.keras.Model):
    def __init__(self, unique_tutors, unique_subjects, unique_grades, unique_teaching_times):
        super().__init__()
        
        unique_tutors = [str(x) for x in unique_tutors]
        unique_subjects = [str(x) for x in unique_subjects]
        unique_grades = [str(x) for x in unique_grades]
        unique_teaching_times = [str(x) for x in unique_teaching_times]
        
        self.tutor_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_tutors, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.tutor_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_tutors) + 2, output_dim=32
        )
        
        self.subject_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_subjects, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.subject_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_subjects) + 2, output_dim=32
        )
        
        self.grade_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_grades, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.grade_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_grades) + 2, output_dim=32
        )
        
        self.teaching_time_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_teaching_times, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.teaching_time_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_teaching_times) + 2, output_dim=32
        )
        
        self.experience_dense = tf.keras.layers.Dense(32)
        
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(160,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32)
        ])
        
        self.candidate_embeddings = None

    def call(self, inputs):
        # Debug inputs
        print("TutorModel input dtypes:", {k: v.dtype for k, v in inputs.items()})
        
        # Ensure inputs are properly typed
        experience = tf.cast(inputs['Kinh nghiệm giảng dạy'], tf.float32)
        experience_2d = tf.expand_dims(experience, axis=-1)
        
        # Convert strings to indices
        tutor_indices = self.tutor_lookup(inputs['Tên gia sư'])
        subject_indices = self.subject_lookup(inputs['Môn học'])
        grade_indices = self.grade_lookup(inputs['Khối Lớp'])
        teaching_time_indices = self.teaching_time_lookup(inputs['Thời gian dạy học'])
        
        # Get embeddings
        tutor_embedding = self.tutor_embedding(tutor_indices)
        subject_embedding = self.subject_embedding(subject_indices)
        grade_embedding = self.grade_embedding(grade_indices)
        teaching_time_embedding = self.teaching_time_embedding(teaching_time_indices)
        experience_embedding = self.experience_dense(experience_2d)
        
        # Debug embedding shapes
        print("Tutor embedding shape:", tutor_embedding.shape)
        print("Subject embedding shape:", subject_embedding.shape)
        print("Grade embedding shape:", grade_embedding.shape)
        print("Teaching time embedding shape:", teaching_time_embedding.shape)
        print("Experience embedding shape:", experience_embedding.shape)
        
        # Concatenate embeddings
        concatenated = tf.concat([
            tutor_embedding, subject_embedding, grade_embedding,
            teaching_time_embedding, experience_embedding
        ], axis=-1)
        print("Concatenated shape:", concatenated.shape)
        
        # Process through dense layers
        output = self.dense_layers(concatenated)
        print("Output shape:", output.shape)
        
        return output

class MaterialModel(tf.keras.Model):
    def __init__(self, unique_materials, unique_subjects, unique_grades, unique_types):
        super().__init__()
        
        unique_materials = [str(x) for x in unique_materials]
        unique_subjects = [str(x) for x in unique_subjects]
        unique_grades = [str(x) for x in unique_grades]
        unique_types = [str(x) for x in unique_types]
        
        self.material_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_materials, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.material_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_materials) + 2, output_dim=32
        )
        
        self.subject_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_subjects, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.subject_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_subjects) + 2, output_dim=32
        )
        
        self.grade_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_grades, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.grade_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_grades) + 2, output_dim=32
        )
        
        self.type_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_types, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.type_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_types) + 2, output_dim=32
        )
        
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(128,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32)
        ])
        
        self.candidate_embeddings = None

    def call(self, inputs):
        # Debug inputs
        print("MaterialModel input dtypes:", {k: v.dtype for k, v in inputs.items()})
        
        # Convert strings to indices
        material_indices = self.material_lookup(inputs['Tên tài liệu'])
        subject_indices = self.subject_lookup(inputs['Môn học'])
        grade_indices = self.grade_lookup(inputs['Khối Lớp'])
        type_indices = self.type_lookup(inputs['Loại tài liệu'])
        
        # Get embeddings
        material_embedding = self.material_embedding(material_indices)
        subject_embedding = self.subject_embedding(subject_indices)
        grade_embedding = self.grade_embedding(grade_indices)
        type_embedding = self.type_embedding(type_indices)
        
        # Debug embedding shapes
        print("Material embedding shape:", material_embedding.shape)
        print("Subject embedding shape:", subject_embedding.shape)
        print("Grade embedding shape:", grade_embedding.shape)
        print("Type embedding shape:", type_embedding.shape)
        
        # Concatenate embeddings
        concatenated = tf.concat([
            material_embedding, subject_embedding,
            grade_embedding, type_embedding
        ], axis=-1)
        print("Concatenated shape:", concatenated.shape)
        
        # Process through dense layers
        output = self.dense_layers(concatenated)
        print("Output shape:", output.shape)
        
        return output

class RecommendationModel(tfrs.Model):
    def __init__(self, student_features, course_features, tutor_features, material_features,
                 student_course_train, student_tutor_train, student_material_train):
        super().__init__()
        
        print("Initializing RecommendationModel...")
        unique_schools = list(set(
            [str(x) for x in student_features['Trường học hiện tại'].unique()] +
            [str(x) for x in student_course_train['Trường học hiện tại'].unique()] +
            [str(x) for x in student_tutor_train['Trường học hiện tại'].unique()] +
            [str(x) for x in student_material_train['Trường học hiện tại'].unique()]
        ))
        unique_grades = list(set(
            [str(x) for x in student_features['Khối Lớp hiện tại'].unique()] +
            [str(x) for x in student_course_train['Khối Lớp hiện tại'].unique()] +
            [str(x) for x in student_tutor_train['Khối Lớp hiện tại'].unique()] +
            [str(x) for x in student_material_train['Khối Lớp hiện tại'].unique()]
        ))
        unique_goals = list(set(
            [str(x) for x in student_features['Mục tiêu học'].unique()] +
            [str(x) for x in student_course_train['Mục tiêu học'].unique()] +
            [str(x) for x in student_tutor_train['Mục tiêu học'].unique()] +
            [str(x) for x in student_material_train['Mục tiêu học'].unique()]
        ))
        unique_favorite_subjects = list(set(
            [str(x) for x in student_features['Môn học yêu thích'].unique()] +
            [str(x) for x in student_course_train['Môn học yêu thích'].unique()] +
            [str(x) for x in student_tutor_train['Môn học yêu thích'].unique()] +
            [str(x) for x in student_material_train['Môn học yêu thích'].unique()]
        ))
        unique_learning_methods = list(set(
            [str(x) for x in student_features['Phương pháp học yêu thích'].unique()] +
            [str(x) for x in student_course_train['Phương pháp học yêu thích'].unique()] +
            [str(x) for x in student_tutor_train['Phương pháp học yêu thích'].unique()] +
            [str(x) for x in student_material_train['Phương pháp học yêu thích'].unique()]
        ))
        
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
        
        # Compute candidate embeddings
        course_embeddings = []
        course_ids = []
        course_dataset = tf.data.Dataset.from_tensor_slices(dict(course_features)).batch(32)
        for batch in course_dataset:
            embeddings = self.course_model(batch)
            ids = tf.strings.join(['course_', tf.strings.as_string(batch['ID Trung Tâm'])])
            course_embeddings.append(embeddings)
            course_ids.append(ids)
            print(f"Course batch embeddings shape: {embeddings.shape}")
            print(f"Course batch IDs shape: {ids.shape}")
        self.course_model.candidate_embeddings = {
            'embeddings': tf.concat(course_embeddings, axis=0),
            'identifiers': tf.concat(course_ids, axis=0)
        }
        print(f"Course embeddings final shape: {self.course_model.candidate_embeddings['embeddings'].shape}")
        print(f"Course IDs final shape: {self.course_model.candidate_embeddings['identifiers'].shape}")
        
        tutor_dataset = tf.data.Dataset.from_tensor_slices({
            k: v for k, v in dict(tutor_features).items()
            if k in ['ID Gia Sư', 'Tên gia sư', 'Môn học', 'Khối Lớp', 'Thời gian dạy học', 'Kinh nghiệm giảng dạy']
        }).batch(32)
        tutor_embeddings = []
        tutor_ids = []
        for batch in tutor_dataset:
            embeddings = self.tutor_model(batch)
            ids = tf.strings.join(['tutor_', tf.strings.as_string(batch['ID Gia Sư'])])
            tutor_embeddings.append(embeddings)
            tutor_ids.append(ids)
            print(f"Tutor batch embeddings shape: {embeddings.shape}")
            print(f"Tutor batch IDs shape: {ids.shape}")
        self.tutor_model.candidate_embeddings = {
            'embeddings': tf.concat(tutor_embeddings, axis=0),
            'identifiers': tf.concat(tutor_ids, axis=0)
        }
        print(f"Tutor embeddings final shape: {self.tutor_model.candidate_embeddings['embeddings'].shape}")
        print(f"Tutor IDs final shape: {self.tutor_model.candidate_embeddings['identifiers'].shape}")
        
        material_embeddings = []
        material_ids = []
        material_dataset = tf.data.Dataset.from_tensor_slices(dict(material_features)).batch(32)
        for batch in material_dataset:
            embeddings = self.material_model(batch)
            ids = tf.strings.join(['material_', tf.strings.as_string(batch['ID Tài Liệu'])])
            material_embeddings.append(embeddings)
            material_ids.append(ids)
            print(f"Material batch embeddings shape: {embeddings.shape}")
            print(f"Material batch IDs shape: {ids.shape}")
        self.material_model.candidate_embeddings = {
            'embeddings': tf.concat(material_embeddings, axis=0),
            'identifiers': tf.concat(material_ids, axis=0)
        }
        print(f"Material embeddings final shape: {self.material_model.candidate_embeddings['embeddings'].shape}")
        print(f"Material IDs final shape: {self.material_model.candidate_embeddings['identifiers'].shape}")
        
        # Combine all candidate embeddings and IDs
        self.all_candidate_embeddings = tf.concat([
            self.course_model.candidate_embeddings['embeddings'],
            self.tutor_model.candidate_embeddings['embeddings'],
            self.material_model.candidate_embeddings['embeddings']
        ], axis=0)
        self.all_candidate_ids = tf.concat([
            self.course_model.candidate_embeddings['identifiers'],
            self.tutor_model.candidate_embeddings['identifiers'],
            self.material_model.candidate_embeddings['identifiers']
        ], axis=0)
        
        self.all_candidate_embeddings = tf.convert_to_tensor(self.all_candidate_embeddings, dtype=tf.float32)
        
        # Ensure all_candidate_ids are valid UTF-8 strings
        self.all_candidate_ids = tf.constant(
            [id_.decode('utf-8') if isinstance(id_, bytes) else str(id_) for id_ in self.all_candidate_ids.numpy()],
            dtype=tf.string,
            shape=self.all_candidate_ids.shape
        )
        
        # Validate candidate IDs
        unique_ids, counts = np.unique(self.all_candidate_ids.numpy(), return_counts=True)
        duplicates = unique_ids[counts > 1]
        if len(duplicates) > 0:
            print(f"Warning: Duplicate IDs found: {duplicates}")
        else:
            print("All candidate IDs are unique.")
        
        print("Checking for invalid IDs...")
        invalid_ids = [id_ for id_ in self.all_candidate_ids.numpy() if not id_ or id_ == 'nan']
        if invalid_ids:
            print(f"Found invalid IDs: {invalid_ids}")
        else:
            print("No invalid IDs found.")
        
        print(f"All candidate embeddings shape: {self.all_candidate_embeddings.shape}")
        print(f"All candidate IDs shape: {self.all_candidate_ids.shape}")
        print(f"Sample candidate IDs: {self.all_candidate_ids[:5].numpy()}")
        print(f"All candidate IDs dtype: {self.all_candidate_ids.dtype}")
        
        # Create StringLookup for candidate IDs
        candidate_id_vocabulary = [id_ for id_ in self.all_candidate_ids.numpy()]
        self.candidate_id_lookup = tf.keras.layers.StringLookup(
            vocabulary=candidate_id_vocabulary,
            mask_token=None,
            output_mode='int',
            num_oov_indices=1,
            oov_token='[UNK]'
        )
        
        # Initialize BruteForce layer
        self.streaming = tfrs.layers.factorized_top_k.BruteForce(k=10)
        self.streaming.index(
            candidates=self.all_candidate_embeddings,
            identifiers=self.all_candidate_ids
        )
        print(f"BruteForce index created. Number of candidates: {self.all_candidate_embeddings.shape[0]}")
        
        # Debug candidate dataset
        candidate_dataset = tf.data.Dataset.from_tensor_slices({
            'embeddings': self.all_candidate_embeddings,
            'identifiers': self.all_candidate_ids
        }).map(lambda x: (x['embeddings'], x['identifiers']))
        for batch in candidate_dataset.batch(32).take(1):
            embeddings, identifiers = batch
            print(f"Candidate dataset sample - embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
            print(f"Candidate dataset sample - identifiers shape: {identifiers.shape}, dtype: {identifiers.dtype}")
        
        # Initialize Retrieval task with BruteForce layer
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=self.streaming,
                ks=[1, 5, 10]
            )
        )
        
        # Store training datasets
        self.course_train_dataset = tf.data.Dataset.from_tensor_slices(dict(student_course_train))
        self.tutor_train_dataset = tf.data.Dataset.from_tensor_slices(dict(student_tutor_train))
        self.material_train_dataset = tf.data.Dataset.from_tensor_slices(dict(student_material_train))

    def call(self, inputs):
        student_embeddings = self.student_model(inputs)
        
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
        print("compute_loss input keys:", list(features.keys()))
        
        student_embeddings = self.student_model(features)
        print(f"Student embeddings shape: {student_embeddings.shape}")
        
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
        tf.print("Sample candidate IDs:", candidate_ids[:5])
        
        # Ensure valid candidate IDs using dynamic shape
        batch_size = tf.shape(candidate_ids)[0]
        unk_tensor = tf.fill([batch_size], tf.constant('[UNK]', dtype=tf.string))
        
        valid_candidate_ids = tf.where(
            tf.reduce_any(tf.equal(candidate_ids[:, None], self.all_candidate_ids[None, :]), axis=-1),
            candidate_ids,
            unk_tensor
        )
        tf.print("Valid candidate IDs:", valid_candidate_ids[:5])
        
        candidate_indices = self.candidate_id_lookup(valid_candidate_ids)
        candidate_indices = tf.cast(candidate_indices, tf.int32)
        candidate_indices = tf.where(
            candidate_indices == tf.constant(0, dtype=tf.int32),
            tf.constant(-1, dtype=tf.int32),
            candidate_indices - tf.constant(1, dtype=tf.int32)
        )
        
        print(f"Candidate indices shape: {candidate_indices.shape}")
        tf.print("Sample candidate indices:", candidate_indices[:5])
        
        valid_mask = candidate_indices >= 0
        valid_indices = tf.boolean_mask(candidate_indices, valid_mask)
        valid_student_embeddings = tf.boolean_mask(student_embeddings, valid_mask)
        valid_candidate_embeddings = tf.boolean_mask(candidate_embeddings, valid_mask)
        valid_candidate_ids = tf.boolean_mask(valid_candidate_ids, valid_mask)
        
        print(f"Valid indices shape: {valid_indices.shape}")
        print(f"Valid student embeddings shape: {valid_student_embeddings.shape}")
        print(f"Valid candidate embeddings shape: {valid_candidate_embeddings.shape}")
        print(f"Valid candidate IDs shape: {valid_candidate_ids.shape}")
        
        if training:
            scores, top_k_ids = self.streaming(valid_student_embeddings, k=10)
            print(f"Streaming scores shape: {scores.shape}")
            print(f"Streaming top_k_ids shape: {top_k_ids.shape}")
            tf.print("Sample top_k_ids:", top_k_ids[:5])
        
        def compute_valid_loss():
            try:
                loss = self.task(
                    query_embeddings=valid_student_embeddings,
                    candidate_embeddings=valid_candidate_embeddings,
                    candidate_ids=valid_candidate_ids,
                    compute_metrics=training
                )
                return loss
            except Exception as e:
                tf.print("Error in task call:", str(e))
                raise
        
        def compute_default_loss():
            tf.print("No valid indices found. Returning default loss.")
            return tf.constant(0.0, dtype=tf.float32)
        
        loss = tf.cond(
            tf.reduce_any(valid_mask),
            compute_valid_loss,
            compute_default_loss
        )
        
        if training:
            for metric in self.task.metrics:
                print(f"Metric {metric.name}: {metric.result()}")
        
        return loss
