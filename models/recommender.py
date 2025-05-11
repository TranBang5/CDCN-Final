import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
from sklearn.utils import resample

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
    
    def create_multi_hot(df, column, prefix, vocab=None):
        if column not in df.columns:
            print(f"Warning: Column {column} not found in DataFrame")
            return df, []
        if vocab is None:
            values = df[column].str.split(',').explode().str.strip().unique()
            vocab = sorted([v for v in values if v and v != 'nan'])
        multi_hot = np.zeros((len(df), len(vocab)), dtype=np.float32)
        for i, row in enumerate(df[column].str.split(',')):
            if isinstance(row, list):
                for val in row:
                    val = val.strip()
                    if val in vocab:
                        multi_hot[i, vocab.index(val)] = 1.0
        multi_hot_df = pd.DataFrame(multi_hot, columns=[f"{prefix}_{v}" for v in vocab], index=df.index)
        print(f"Created multi-hot columns for {column}: {[f'{prefix}_{v}' for v in vocab[:5]]}...")
        return pd.concat([df, multi_hot_df], axis=1), vocab

    # Create vocabularies
    subject_vocab = sorted(set(student_data['Môn học yêu thích'].str.split(',').explode().str.strip()) | 
                         set(course_data['Môn học'].str.split(',').explode().str.strip()) | 
                         set(tutor_data['Môn học'].str.split(',').explode().str.strip()) | 
                         set(material_data['Môn học'].str.split(',').explode().str.strip()))
    grade_vocab = sorted(set(student_data['Khối Lớp hiện tại'].str.replace('Lớp ', '').str.split(',').explode().str.strip()) | 
                        set(course_data['Khối Lớp'].str.split(',').explode().str.strip()) | 
                        set(tutor_data['Khối Lớp'].str.split(',').explode().str.strip()) | 
                        set(material_data['Khối Lớp'].str.split(',').explode().str.strip()))
    material_type_vocab = sorted(set(material_data['Loại tài liệu'].str.split(',').explode().str.strip()))
    
    print(f"Subject vocab: {subject_vocab[:5]}... (total: {len(subject_vocab)})")
    print(f"Grade vocab: {grade_vocab[:5]}... (total: {len(grade_vocab)})")
    print(f"Material type vocab: {material_type_vocab[:5]}... (total: {len(material_type_vocab)})")

    # Apply multi-hot encoding
    for df in [student_data, student_course_train, student_course_test, student_tutor_train, student_tutor_test, student_material_train, student_material_test]:
        df, _ = create_multi_hot(df, 'Môn học yêu thích', 'subject', subject_vocab)
        df, _ = create_multi_hot(df, 'Khối Lớp hiện tại', 'grade', grade_vocab)
    for df in [course_data, student_course_train, student_course_test]:
        df, _ = create_multi_hot(df, 'Môn học', 'subject', subject_vocab)
        df, _ = create_multi_hot(df, 'Khối Lớp', 'grade', grade_vocab)
    for df in [tutor_data, student_tutor_train, student_tutor_test]:
        df, _ = create_multi_hot(df, 'Môn học', 'subject', subject_vocab)
        df, _ = create_multi_hot(df, 'Khối Lớp', 'grade', grade_vocab)
    for df in [material_data, student_material_train, student_material_test]:
        df, _ = create_multi_hot(df, 'Môn học', 'subject', subject_vocab)
        df, _ = create_multi_hot(df, 'Khối Lớp', 'grade', grade_vocab)
        df, _ = create_multi_hot(df, 'Loại tài liệu', 'material_type', material_type_vocab)

    # Debug multi-hot columns
    print("Multi-hot columns in course_data:", [col for col in course_data.columns if col.startswith('subject_') or col.startswith('grade_')][:10])
    print("Multi-hot columns in tutor_data:", [col for col in tutor_data.columns if col.startswith('subject_') or col.startswith('grade_')][:10])
    print("Multi-hot columns in material_data:", [col for col in material_data.columns if col.startswith('subject_') or col.startswith('grade_') or col.startswith('material_type_')][:10])

    minority_ids = student_tutor_train['ID Gia Sư'].value_counts()[student_tutor_train['ID Gia Sư'].value_counts() < 5].index
    for id_ in minority_ids:
        minority_df = student_tutor_train[student_tutor_train['ID Gia Sư'] == id_]
        upsampled = resample(minority_df, replace=True, n_samples=5, random_state=42)
        student_tutor_train = pd.concat([student_tutor_train, upsampled])

    valid_goals = set(student_data['Mục tiêu học'].astype(str))
    for df in [student_course_train, student_tutor_train, student_material_train]:
        df['Mục tiêu học'] = df['Mục tiêu học'].apply(lambda x: str(x) if str(x) in valid_goals else 'cat_unknown')
    valid_subjects = set(student_data['Môn học yêu thích'].astype(str))
    for df in [student_course_train, student_tutor_train, student_material_train]:
        df['Môn học yêu thích'] = df['Môn học yêu thích'].apply(lambda x: str(x) if str(x) in valid_subjects else 'cat_unknown')

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
        'Trường học hiện tại', 'Mục tiêu học', 'Phương pháp học yêu thích',
        'Tên Trung Tâm', 'Phương pháp học', 'Tên gia sư', 'Thời gian dạy học',
        'Tên tài liệu', 'Địa chỉ'
    ]
    
    multi_hot_columns = [
        col for col in student_data.columns if col.startswith('subject_') or col.startswith('grade_') or col.startswith('material_type_')
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
        for col in multi_hot_columns[:5]:
            if col in df.columns:
                print(f"{col}: {df[col].iloc[0]}")
    
    def create_tf_dataset(df):
        dtype_dict = {}
        for col in df.columns:
            if col in categorical_columns or col in id_columns or col in ['Tên']:
                dtype_dict[col] = tf.string
            elif col in numeric_columns or col in multi_hot_columns:
                dtype_dict[col] = tf.float32 if numeric_columns.get(col, 'float32') == 'float32' else tf.float64
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
        'student_material_train_dataset': student_material_train_dataset,
        'subject_vocab': subject_vocab,
        'grade_vocab': grade_vocab,
        'material_type_vocab': material_type_vocab
    }

class StudentTower(tf.keras.Model):
    def __init__(self, unique_schools, unique_goals, unique_learning_methods, subject_vocab, grade_vocab):
        super().__init__()
        
        unique_schools = [str(x) for x in unique_schools]
        unique_goals = [str(x) for x in unique_goals]
        unique_learning_methods = [str(x) for x in unique_learning_methods]
        self.subject_vocab = [str(x) for x in subject_vocab]
        self.grade_vocab = [str(x) for x in grade_vocab]
        
        print(f"Unique schools: {len(unique_schools)}, Sample: {unique_schools[:5]}")
        print(f"Unique goals: {len(unique_goals)}, Sample: {unique_goals[:5]}")
        print(f"Unique learning methods: {len(unique_learning_methods)}, Sample: {unique_learning_methods[:5]}")
        print(f"Subject vocab size: {len(self.subject_vocab)}, Sample: {self.subject_vocab[:5]}")
        print(f"Grade vocab size: {len(self.grade_vocab)}, Sample: {self.grade_vocab[:5]}")
        
        self.school_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_schools, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.school_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_schools) + 2, output_dim=32
        )
        
        self.goal_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_goals, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.goal_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_goals) + 2, output_dim=32
        )
        
        self.learning_method_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_learning_methods, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.learning_method_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_learning_methods) + 2, output_dim=32
        )
        
        self.subject_dense = tf.keras.layers.Dense(32, input_shape=(len(self.subject_vocab),))
        self.grade_dense = tf.keras.layers.Dense(32, input_shape=(len(self.grade_vocab),))
        
        input_dim = 32 * 3 + 32 + 32
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.005), input_shape=(input_dim,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32)
        ])

    def call(self, inputs):
        print("StudentTower input dtypes:", {k: v.dtype for k, v in inputs.items()})
        
        school_indices = self.school_lookup(inputs['Trường học hiện tại'])
        goal_indices = self.goal_lookup(inputs['Mục tiêu học'])
        learning_method_indices = self.learning_method_lookup(inputs['Phương pháp học yêu thích'])
        
        school_embedding = tf.keras.layers.Dropout(0.3)(self.school_embedding(school_indices))
        goal_embedding = self.goal_embedding(goal_indices)
        learning_method_embedding = self.learning_method_embedding(learning_method_indices)
        
        subject_inputs = [inputs.get(f'subject_{s}', tf.zeros_like(inputs['Mục tiêu học'], dtype=tf.float32)) for s in self.subject_vocab]
        subject_multi_hot = tf.stack(subject_inputs, axis=-1)
        subject_embedding = self.subject_dense(subject_multi_hot)
        
        grade_inputs = [inputs.get(f'grade_{g}', tf.zeros_like(inputs['Mục tiêu học'], dtype=tf.float32)) for g in self.grade_vocab]
        grade_multi_hot = tf.stack(grade_inputs, axis=-1)
        grade_embedding = self.grade_dense(grade_multi_hot)
        
        print("School embedding shape:", school_embedding.shape)
        print("Goal embedding shape:", goal_embedding.shape)
        print("Learning method embedding shape:", learning_method_embedding.shape)
        print("Subject embedding shape:", subject_embedding.shape)
        print("Grade embedding shape:", grade_embedding.shape)
        
        concatenated = tf.concat([
            school_embedding, goal_embedding, learning_method_embedding,
            subject_embedding, grade_embedding
        ], axis=-1)
        print("Concatenated shape:", concatenated.shape)
        
        output = self.dense_layers(concatenated)
        print("Output shape:", output.shape)
        
        return output

class CourseModel(tf.keras.Model):
    def __init__(self, unique_centers, unique_methods, subject_vocab, grade_vocab):
        super().__init__()
        
        unique_centers = [str(x) for x in unique_centers]
        unique_methods = [str(x) for x in unique_methods]
        self.subject_vocab = [str(x) for x in subject_vocab]
        self.grade_vocab = [str(x) for x in grade_vocab]
        
        self.center_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_centers, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.center_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_centers) + 2, output_dim=32
        )
        
        self.method_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_methods, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.method_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_methods) + 2, output_dim=32
        )
        
        self.subject_dense = tf.keras.layers.Dense(32, input_shape=(len(self.subject_vocab),))
        self.grade_dense = tf.keras.layers.Dense(32, input_shape=(len(self.grade_vocab),))
        
        self.cost_dense = tf.keras.layers.Dense(32)
        self.time_dense = tf.keras.layers.Dense(32)
        
        input_dim = 32 * 2 + 32 + 32 + 32 + 32
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.005), input_shape=(input_dim,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.005)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32)
        ])
        
        self.candidate_embeddings = None

    def call(self, inputs):
        print("CourseModel input keys:", list(inputs.keys()))
        print("CourseModel input dtypes:", {k: v.dtype for k, v in inputs.items()})
        
        cost = tf.cast(inputs.get('Chi phí', tf.zeros_like(inputs['Tên Trung Tâm'], dtype=tf.float32)), tf.float32)
        time = tf.cast(inputs.get('Thời gian', tf.zeros_like(inputs['Tên Trung Tâm'], dtype=tf.float32)), tf.float32)
        
        cost_2d = tf.expand_dims(cost, axis=-1)
        time_2d = tf.expand_dims(time, axis=-1)
        
        center_indices = self.center_lookup(inputs['Tên Trung Tâm'])
        method_indices = self.method_lookup(inputs['Phương pháp học'])
        
        center_embedding = self.center_embedding(center_indices)
        method_embedding = self.method_embedding(method_indices)
        
        subject_inputs = [inputs.get(f'subject_{s}', tf.zeros_like(inputs['Tên Trung Tâm'], dtype=tf.float32)) for s in self.subject_vocab]
        subject_multi_hot = tf.stack(subject_inputs, axis=-1)
        subject_embedding = self.subject_dense(subject_multi_hot)
        
        grade_inputs = [inputs.get(f'grade_{g}', tf.zeros_like(inputs['Tên Trung Tâm'], dtype=tf.float32)) for g in self.grade_vocab]
        grade_multi_hot = tf.stack(grade_inputs, axis=-1)
        grade_embedding = self.grade_dense(grade_multi_hot)
        
        cost_embedding = self.cost_dense(cost_2d)
        time_embedding = self.time_dense(time_2d)
        
        print("Center embedding shape:", center_embedding.shape)
        print("Method embedding shape:", method_embedding.shape)
        print("Subject embedding shape:", subject_embedding.shape)
        print("Grade embedding shape:", grade_embedding.shape)
        print("Cost embedding shape:", cost_embedding.shape)
        print("Time embedding shape:", time_embedding.shape)
        
        concatenated = tf.concat([
            center_embedding, method_embedding, subject_embedding,
            grade_embedding, cost_embedding, time_embedding
        ], axis=-1)
        print("Concatenated shape:", concatenated.shape)
        
        output = self.dense_layers(concatenated)
        print("Output shape:", output.shape)
        
        return output

class TutorModel(tf.keras.Model):
    def __init__(self, unique_tutors, unique_teaching_times, subject_vocab, grade_vocab):
        super().__init__()
        
        unique_tutors = [str(x) for x in unique_tutors]
        unique_teaching_times = [str(x) for x in unique_teaching_times]
        self.subject_vocab = [str(x) for x in subject_vocab]
        self.grade_vocab = [str(x) for x in grade_vocab]
        
        self.tutor_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_tutors, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.tutor_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_tutors) + 2, output_dim=32
        )
        
        self.teaching_time_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_teaching_times, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.teaching_time_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_teaching_times) + 2, output_dim=32
        )
        
        self.subject_dense = tf.keras.layers.Dense(32, input_shape=(len(self.subject_vocab),))
        self.grade_dense = tf.keras.layers.Dense(32, input_shape=(len(self.grade_vocab),))
        
        self.experience_dense = tf.keras.layers.Dense(32)
        
        input_dim = 32 * 2 + 32 + 32 + 32
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.005), input_shape=(input_dim,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.005)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32)
        ])
        
        self.candidate_embeddings = None

    def call(self, inputs):
        print("TutorModel input keys:", list(inputs.keys()))
        print("TutorModel input dtypes:", {k: v.dtype for k, v in inputs.items()})
        
        experience = tf.cast(inputs.get('Kinh nghiệm giảng dạy', tf.zeros_like(inputs['Tên gia sư'], dtype=tf.float32)), tf.float32)
        experience_2d = tf.expand_dims(experience, axis=-1)
        
        tutor_indices = self.tutor_lookup(inputs['Tên gia sư'])
        teaching_time_indices = self.teaching_time_lookup(inputs['Thời gian dạy học'])
        
        tutor_embedding = self.tutor_embedding(tutor_indices)
        teaching_time_embedding = self.teaching_time_embedding(teaching_time_indices)
        
        subject_inputs = [inputs.get(f'subject_{s}', tf.zeros_like(inputs['Tên gia sư'], dtype=tf.float32)) for s in self.subject_vocab]
        subject_multi_hot = tf.stack(subject_inputs, axis=-1)
        subject_embedding = self.subject_dense(subject_multi_hot)
        
        grade_inputs = [inputs.get(f'grade_{g}', tf.zeros_like(inputs['Tên gia sư'], dtype=tf.float32)) for g in self.grade_vocab]
        grade_multi_hot = tf.stack(grade_inputs, axis=-1)
        grade_embedding = self.grade_dense(grade_multi_hot)
        
        experience_embedding = self.experience_dense(experience_2d)
        
        print("Tutor embedding shape:", tutor_embedding.shape)
        print("Teaching time embedding shape:", teaching_time_embedding.shape)
        print("Subject embedding shape:", subject_embedding.shape)
        print("Grade embedding shape:", grade_embedding.shape)
        print("Experience embedding shape:", experience_embedding.shape)
        
        concatenated = tf.concat([
            tutor_embedding, teaching_time_embedding, subject_embedding,
            grade_embedding, experience_embedding
        ], axis=-1)
        print("Concatenated shape:", concatenated.shape)
        
        output = self.dense_layers(concatenated)
        print("Output shape:", output.shape)
        
        return output

class MaterialModel(tf.keras.Model):
    def __init__(self, unique_materials, subject_vocab, grade_vocab, material_type_vocab):
        super().__init__()
        
        unique_materials = [str(x) for x in unique_materials]
        self.subject_vocab = [str(x) for x in subject_vocab]
        self.grade_vocab = [str(x) for x in grade_vocab]
        self.material_type_vocab = [str(x) for x in material_type_vocab]
        
        self.material_lookup = tf.keras.layers.StringLookup(
            vocabulary=unique_materials, mask_token=None, output_mode='int',
            num_oov_indices=1, oov_token='[UNK]'
        )
        self.material_embedding = tf.keras.layers.Embedding(
            input_dim=len(unique_materials) + 2, output_dim=32
        )
        
        self.subject_dense = tf.keras.layers.Dense(32, input_shape=(len(self.subject_vocab),))
        self.grade_dense = tf.keras.layers.Dense(32, input_shape=(len(self.grade_vocab),))
        self.type_dense = tf.keras.layers.Dense(32, input_shape=(len(self.material_type_vocab),))
        
        input_dim = 32 + 32 + 32 + 32
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.005), input_shape=(input_dim,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.005)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32)
        ])
        
        self.candidate_embeddings = None

    def call(self, inputs):
        print("MaterialModel input keys:", list(inputs.keys()))
        print("MaterialModel input dtypes:", {k: v.dtype for k, v in inputs.items()})
        
        material_indices = self.material_lookup(inputs['Tên tài liệu'])
        
        material_embedding = self.material_embedding(material_indices)
        
        subject_inputs = [inputs.get(f'subject_{s}', tf.zeros_like(inputs['Tên tài liệu'], dtype=tf.float32)) for s in self.subject_vocab]
        subject_multi_hot = tf.stack(subject_inputs, axis=-1)
        subject_embedding = self.subject_dense(subject_multi_hot)
        
        grade_inputs = [inputs.get(f'grade_{g}', tf.zeros_like(inputs['Tên tài liệu'], dtype=tf.float32)) for g in self.grade_vocab]
        grade_multi_hot = tf.stack(grade_inputs, axis=-1)
        grade_embedding = self.grade_dense(grade_multi_hot)
        
        type_inputs = [inputs.get(f'material_type_{t}', tf.zeros_like(inputs['Tên tài liệu'], dtype=tf.float32)) for t in self.material_type_vocab]
        type_multi_hot = tf.stack(type_inputs, axis=-1)
        type_embedding = self.type_dense(type_multi_hot)
        
        print("Material embedding shape:", material_embedding.shape)
        print("Subject embedding shape:", subject_embedding.shape)
        print("Grade embedding shape:", grade_embedding.shape)
        print("Type embedding shape:", type_embedding.shape)
        
        concatenated = tf.concat([
            material_embedding, subject_embedding,
            grade_embedding, type_embedding
        ], axis=-1)
        print("Concatenated shape:", concatenated.shape)
        
        output = self.dense_layers(concatenated)
        print("Output shape:", output.shape)
        
        return output

class RecommendationModel(tfrs.Model):
    def __init__(self, student_features, course_features, tutor_features, material_features,
                 student_course_train, student_tutor_train, student_material_train,
                 subject_vocab, grade_vocab, material_type_vocab):
        super().__init__()
        
        print("Initializing RecommendationModel...")
        unique_schools = list(set(
            [str(x) for x in student_features['Trường học hiện tại'].unique()] +
            [str(x) for x in student_course_train['Trường học hiện tại'].unique()] +
            [str(x) for x in student_tutor_train['Trường học hiện tại'].unique()] +
            [str(x) for x in student_material_train['Trường học hiện tại'].unique()]
        ))
        unique_goals = list(set(
            [str(x) for x in student_features['Mục tiêu học'].unique()] +
            [str(x) for x in student_course_train['Mục tiêu học'].unique()] +
            [str(x) for x in student_tutor_train['Mục tiêu học'].unique()] +
            [str(x) for x in student_material_train['Mục tiêu học'].unique()]
        ))
        unique_learning_methods = list(set(
            [str(x) for x in student_features['Phương pháp học yêu thích'].unique()] +
            [str(x) for x in student_course_train['Phương pháp học yêu thích'].unique()] +
            [str(x) for x in student_tutor_train['Phương pháp học yêu thích'].unique()] +
            [str(x) for x in student_material_train['Phương pháp học yêu thích'].unique()]
        ))
        
        unique_centers = [str(x) for x in course_features['Tên Trung Tâm'].unique()]
        unique_course_methods = [str(x) for x in course_features['Phương pháp học'].unique()]
        
        unique_tutors = [str(x) for x in tutor_features['Tên gia sư'].unique()]
        unique_teaching_times = [str(x) for x in tutor_features['Thời gian dạy học'].unique()]
        
        unique_materials = [str(x) for x in material_features['Tên tài liệu'].unique()]
        
        self.student_model = StudentTower(
            unique_schools, unique_goals, unique_learning_methods,
            subject_vocab, grade_vocab
        )
        
        self.course_model = CourseModel(
            unique_centers, unique_course_methods,
            subject_vocab, grade_vocab
        )
        
        self.tutor_model = TutorModel(
            unique_tutors, unique_teaching_times,
            subject_vocab, grade_vocab
        )
        
        self.material_model = MaterialModel(
            unique_materials, subject_vocab, grade_vocab, material_type_vocab
        )
        
        # Compute candidate embeddings
        course_columns = [
            'ID Trung Tâm', 'Tên Trung Tâm', 'Phương pháp học', 'Thời gian', 'Chi phí', 'Địa chỉ', 'Đánh giá'
        ] + [f'subject_{s}' for s in subject_vocab] + [f'grade_{g}' for g in grade_vocab]
        print("Course columns for candidate dataset:", course_columns)
        # Filter out columns that don't exist
        available_course_columns = [col for col in course_columns if col in course_features.columns]
        print("Available course columns:", available_course_columns)
        course_dataset = tf.data.Dataset.from_tensor_slices({
            k: v for k, v in dict(course_features).items() if k in available_course_columns
        }).batch(32)
        course_embeddings = []
        course_ids = []
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
        
        tutor_columns = [
            'ID Gia Sư', 'Tên gia sư', 'Thời gian dạy học', 'Kinh nghiệm giảng dạy'
        ] + [f'subject_{s}' for s in subject_vocab] + [f'grade_{g}' for g in grade_vocab]
        print("Tutor columns for candidate dataset:", tutor_columns)
        available_tutor_columns = [col for col in tutor_columns if col in tutor_features.columns]
        print("Available tutor columns:", available_tutor_columns)
        tutor_dataset = tf.data.Dataset.from_tensor_slices({
            k: v for k, v in dict(tutor_features).items() if k in available_tutor_columns
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
        
        material_columns = [
            'ID Tài Liệu', 'Tên tài liệu'
        ] + [f'subject_{s}' for s in subject_vocab] + [f'grade_{g}' for g in grade_vocab] + [f'material_type_{t}' for t in material_type_vocab]
        print("Material columns for candidate dataset:", material_columns)
        available_material_columns = [col for col in material_columns if col in material_features.columns]
        print("Available material columns:", available_material_columns)
        material_dataset = tf.data.Dataset.from_tensor_slices({
            k: v for k, v in dict(material_features).items() if k in available_material_columns
        }).batch(32)
        material_embeddings = []
        material_ids = []
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
        
        self.all_candidate_ids = tf.constant(
            [id_.decode('utf-8') if isinstance(id_, bytes) else str(id_) for id_ in self.all_candidate_ids.numpy()],
            dtype=tf.string,
            shape=self.all_candidate_ids.shape
        )
        
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
        
        candidate_id_vocabulary = [id_ for id_ in self.all_candidate_ids.numpy()]
        self.candidate_id_lookup = tf.keras.layers.StringLookup(
            vocabulary=candidate_id_vocabulary,
            mask_token=None,
            output_mode='int',
            num_oov_indices=1,
            oov_token='[UNK]'
        )
        
        self.streaming = tfrs.layers.factorized_top_k.BruteForce(k=10)
        self.streaming.index(
            candidates=self.all_candidate_embeddings,
            identifiers=self.all_candidate_ids
        )
        print(f"BruteForce index created. Number of candidates: {self.all_candidate_embeddings.shape[0]}")
        
        candidate_dataset = tf.data.Dataset.from_tensor_slices({
            'embeddings': self.all_candidate_embeddings,
            'identifiers': self.all_candidate_ids
        }).map(lambda x: (x['embeddings'], x['identifiers']))
        for batch in candidate_dataset.batch(32).take(1):
            embeddings, identifiers = batch
            print(f"Candidate dataset sample - embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
            print(f"Candidate dataset sample - identifiers shape: {identifiers.shape}, dtype: {identifiers.dtype}")
        
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=self.streaming,
                ks=[1, 5, 10]
            )
        )
        
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