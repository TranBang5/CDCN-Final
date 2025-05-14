import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.decomposition import PCA
import logging
import re
import os

# Suppress TensorFlow warnings for cleaner logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_tf_dataset(df):
    dtype_dict = {}
    categorical_columns = [
        'Trường học hiện tại', 'Mục tiêu học', 'Phương pháp học yêu thích',
        'Tên Trung Tâm', 'Phương pháp học', 'Tên gia sư', 'Thời gian dạy học',
        'Tên tài liệu', 'Địa chỉ', 'Khối Lớp', 'Khối Lớp hiện tại',
        'Loại tài liệu', 'Môn học', 'Môn học yêu thích'
    ]
    id_columns = ['ID Học Sinh', 'ID Trung Tâm', 'ID Gia Sư', 'ID Tài Liệu']
    numeric_columns = {
        'Chi phí': 'float32',
        'Thời gian': 'float32',
        'Kinh nghiệm giảng dạy': 'float32',
        'Đánh giá': 'float32'
    }
    multi_hot_columns = [col for col in df.columns if col.startswith('subject_') or col.startswith('grade_') or col.startswith('material_type_')]
    
    for col in df.columns:
        if col in categorical_columns or col in id_columns or col in ['Tên']:
            dtype_dict[col] = tf.string
        elif col in numeric_columns or col in multi_hot_columns or col.startswith('subject_pca_') or col.startswith('grade_pca_') or col.startswith('material_type_pca_'):
            dtype_dict[col] = tf.float32
        else:
            dtype_dict[col] = tf.string
            logger.warning(f"Column {col} not explicitly typed, defaulting to string")
    
    # Preprocess the DataFrame
    for col in df.columns:
        if col in categorical_columns or col in id_columns or col in ['Tên']:
            df[col] = df[col].fillna('unknown').astype(str)
            logger.debug(f"Column {col}: unique values = {df[col].unique()[:5]}")
        elif col in numeric_columns or col in multi_hot_columns or col.startswith('subject_pca_') or col.startswith('grade_pca_') or col.startswith('material_type_pca_'):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float32)
        else:
            logger.warning(f"Column {col} not preprocessed, defaulting to string")
            df[col] = df[col].fillna('unknown').astype(str)
    
    data_dict = {col: df[col].values for col in df.columns}
    for col, dtype in dtype_dict.items():
        if col not in data_dict:
            logger.warning(f"Column {col} missing, filling with zeros")
            data_dict[col] = np.zeros(len(df), dtype=np.bytes_ if dtype == tf.string else np.float32)
        elif dtype == tf.string:
            data_dict[col] = np.array([
                x.encode('utf-8') if isinstance(x, str) and x else b'unknown'
                for x in data_dict[col]
            ], dtype=np.bytes_)
        else:
            data_dict[col] = np.array(data_dict[col], dtype=np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices(data_dict)
    return dataset

def load_and_preprocess_data():
    categorical_dtypes = {col: str for col in [
        'Trường học hiện tại', 'Khối Lớp hiện tại', 'Mục tiêu học',
        'Môn học yêu thích', 'Phương pháp học yêu thích', 'Tên Trung Tâm',
        'Môn học', 'Khối Lớp', 'Phương pháp học', 'Tên gia sư',
        'Thời gian dạy học', 'Tên tài liệu', 'Loại tài liệu', 'Địa chỉ'
    ]}
    
    logger.info("Loading CSV files...")
    student_data = pd.read_csv('data/hoc_sinh.csv', dtype=categorical_dtypes, encoding='utf-8')
    course_data = pd.read_csv('data/trung_tam.csv', dtype=categorical_dtypes, encoding='utf-8')
    tutor_data = pd.read_csv('data/gia_su.csv', dtype=categorical_dtypes, encoding='utf-8')
    material_data = pd.read_csv('data/tai_lieu.csv', dtype=categorical_dtypes, encoding='utf-8')
    
    # Validate grade columns
    for df, name in [(student_data, 'student_data'), (course_data, 'course_data')]:
        for col in ['Khối Lớp hiện tại', 'Khối Lớp']:
            if col in df.columns:
                unique_grades = df[col].unique()
                logger.info(f"{name} - {col}: unique values = {unique_grades}")
                if len(unique_grades) <= 1:
                    logger.warning(f"{name} - {col} has insufficient unique values: {unique_grades}")
    
    if 'Phương pháp dạy' in tutor_data.columns:
        tutor_data = tutor_data.drop(columns=['Phương pháp dạy'])
    
    if 'Thời gian dạy học' in tutor_data.columns:
        tutor_data['Thời gian dạy học'] = tutor_data['Thời gian dạy học'].apply(
            lambda x: ','.join(x.split()) if isinstance(x, str) else x
        )
    
    if 'Kinh nghiệm giảng dạy' in tutor_data.columns:
        def extract_years(value):
            if pd.isna(value) or not isinstance(value, str):
                return 0.0
            match = re.match(r'(\d+\.?\d*)\s*năm', value.strip(), re.IGNORECASE)
            return float(match.group(1)) if match else 0.0
        tutor_data['Kinh nghiệm giảng dạy'] = tutor_data['Kinh nghiệm giảng dạy'].apply(extract_years)
    
    student_course_train = pd.read_csv('data/hoc_sinh_trung_tam_train.csv', dtype=categorical_dtypes, encoding='utf-8')
    student_course_test = pd.read_csv('data/hoc_sinh_trung_tam_test.csv', dtype=categorical_dtypes, encoding='utf-8')
    student_tutor_train = pd.read_csv('data/hoc_sinh_gia_su_train.csv', dtype=categorical_dtypes, encoding='utf-8')
    student_tutor_test = pd.read_csv('data/hoc_sinh_gia_su_test.csv', dtype=categorical_dtypes, encoding='utf-8')
    student_material_train = pd.read_csv('data/hoc_sinh_tai_lieu_train.csv', dtype=categorical_dtypes, encoding='utf-8')
    student_material_test = pd.read_csv('data/hoc_sinh_tai_lieu_test.csv', dtype=categorical_dtypes, encoding='utf-8')
    
    # Validate IDs
    logger.info("Validating IDs...")
    course_ids = set(course_data['ID Trung Tâm'].astype(str))
    tutor_ids = set(tutor_data['ID Gia Sư'].astype(str))
    material_ids = set(material_data['ID Tài Liệu'].astype(str))
    
    for df, name, id_col in [
        (student_course_train, 'student_course_train', 'ID Trung Tâm'),
        (student_course_test, 'student_course_test', 'ID Trung Tâm'),
        (student_tutor_train, 'student_tutor_train', 'ID Gia Sư'),
        (student_tutor_test, 'student_tutor_test', 'ID Gia Sư'),
        (student_material_train, 'student_material_train', 'ID Tài Liệu'),
        (student_material_test, 'student_material_test', 'ID Tài Liệu')
    ]:
        valid_ids = course_ids if id_col == 'ID Trung Tâm' else tutor_ids if id_col == 'ID Gia Sư' else material_ids
        mask = df[id_col].astype(str).isin(valid_ids)
        missing_ids = set(df[~mask][id_col].astype(str))
        if missing_ids:
            logger.warning(f"Removing {len(df[~mask])} rows with invalid {id_col} from {name}: {missing_ids}")
            df.drop(df[~mask].index, inplace=True)
            if name == 'student_course_train':
                student_course_train = df
            elif name == 'student_course_test':
                student_course_test = df
            elif name == 'student_tutor_train':
                student_tutor_train = df
            elif name == 'student_tutor_test':
                student_tutor_test = df
            elif name == 'student_material_train':
                student_material_train = df
            elif name == 'student_material_test':
                student_material_test = df
    
    def align_features(df, target_columns):
        for col in target_columns:
            if col not in df.columns:
                df[col] = 0.0 if col in ['Chi phí', 'Thời gian', 'Kinh nghiệm giảng dạy', 'Đánh giá'] else 'unknown'
        return df[target_columns]

    target_columns = list(set(
        student_course_train.columns
        ).union(
        student_tutor_train.columns,
        student_material_train.columns
    ))
    student_course_train = align_features(student_course_train, target_columns)
    student_tutor_train = align_features(student_tutor_train, target_columns)
    student_material_train = align_features(student_material_train, target_columns)

    def create_multi_hot(df, column, prefix, vocab=None):
        if column not in df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            return df, []
        if vocab is None:
            values = df[column].astype(str).str.split(',').explode().str.strip().unique()
            vocab = sorted([v for v in values if v and v != 'nan' and v != '0.0' and not v.replace('.', '').isdigit()])
        multi_hot = np.zeros((len(df), len(vocab)), dtype=np.float32)
        for i, row in enumerate(df[column].astype(str)):
            if row == 'nan' or not row or row == '0.0' or row.replace('.', '').isdigit():
                if column == 'Loại tài liệu':
                    values = ['Điện tử']
                else:
                    continue
            else:
                values = [v.strip() for v in row.split(',') if v.strip()]
            normalized_values = []
            for v in values:
                if v.isdigit():
                    normalized_values.append(f'Lớp {v}')
                elif v in ['Giấy', 'Điện tử']:
                    normalized_values.append(v)
                else:
                    normalized_values.append(v)
            for val in normalized_values:
                if val in vocab:
                    multi_hot[i, vocab.index(val)] = 1.0
        multi_hot_df = pd.DataFrame(multi_hot, columns=[f"{prefix}_{v}" for v in vocab], index=df.index)
        return pd.concat([df, multi_hot_df], axis=1), vocab
    
    def consolidate_vocab(df_list, column):
        values = set()
        for df in df_list:
            if column in df.columns:
                exploded = df[column].astype(str).str.split(',').explode().str.strip()
                for v in exploded:
                    if v and v != 'nan' and v != '0.0' and not v.replace('.', '').isdigit():
                        if v.isdigit():
                            values.add(f'Lớp {v}')
                        elif v in ['Giấy', 'Điện tử']:
                            values.add(v)
                        else:
                            values.add(v)
        return sorted(list(values))

    subject_vocab = consolidate_vocab([student_data, course_data, tutor_data, material_data, 
                                      student_course_train, student_course_test, 
                                      student_tutor_train, student_tutor_test, 
                                      student_material_train, student_material_test], 
                                     'Môn học yêu thích')
    grade_vocab = consolidate_vocab([student_data, course_data, tutor_data, material_data, 
                                    student_course_train, student_course_test, 
                                    student_tutor_train, student_tutor_test, 
                                    student_material_train, student_material_test], 
                                   'Khối Lớp hiện tại')
    material_type_vocab = consolidate_vocab([material_data, student_material_train, student_material_test], 
                                           'Loại tài liệu')
    teaching_time_vocab = consolidate_vocab([tutor_data, student_tutor_train, student_tutor_test], 
                                           'Thời gian dạy học')

    def apply_pca(df, prefix, n_components=10, skip_if_low_variance=True):
        multi_hot_cols = [col for col in df.columns if col.startswith(prefix)]
        if not multi_hot_cols:
            logger.warning(f"No multi-hot columns found for prefix {prefix}")
            desired_components = 10 if prefix == 'subject' else 5
            pca_cols = [f"{prefix}_pca_{i}" for i in range(desired_components)]
            df[pca_cols] = np.zeros((len(df), desired_components), dtype=np.float32)
            return df, pca_cols
        n_samples, n_features = df[multi_hot_cols].shape
        std_sum = df[multi_hot_cols].std().sum()
        unique_counts = {col: len(df[col].unique()) for col in multi_hot_cols}
        logger.info(f"PCA for {prefix}: n_samples={n_samples}, n_features={n_features}, std_sum={std_sum}, unique_counts={unique_counts}")
        if skip_if_low_variance and std_sum < 1e-10:
            logger.warning(f"Skipping PCA for {prefix}: zero or near-zero variance, using raw features")
            return df, multi_hot_cols
        
        n_components = min(n_components, n_samples, n_features)
        pca = PCA(n_components=n_components, svd_solver='full')
        pca_features = pca.fit_transform(df[multi_hot_cols])
        pca_cols = [f"{prefix}_pca_{i}" for i in range(pca_features.shape[1])]
        df = df.drop(columns=multi_hot_cols, errors='ignore')
        df[pca_cols] = pca_features
        desired_components = 10 if prefix == 'subject' else 5
        if pca_features.shape[1] < desired_components:
            padding = np.zeros((len(df), desired_components - pca_features.shape[1]))
            df[[f"{prefix}_pca_{i}" for i in range(pca_features.shape[1], desired_components)]] = padding
            pca_cols.extend([f"{prefix}_pca_{i}" for i in range(pca_features.shape[1], desired_components)])
        return df, pca_cols

    dataframes = [
        student_data, course_data, tutor_data, material_data,
        student_course_train, student_course_test,
        student_tutor_train, student_tutor_test,
        student_material_train, student_material_test
    ]
    processed_dataframes = []
    for df in dataframes:
        if 'Môn học yêu thích' in df.columns:
            df, _ = create_multi_hot(df, 'Môn học yêu thích', 'subject', subject_vocab)
        elif 'Môn học' in df.columns:
            df, _ = create_multi_hot(df, 'Môn học', 'subject', subject_vocab)
        if 'Khối Lớp hiện tại' in df.columns:
            df, _ = create_multi_hot(df, 'Khối Lớp hiện tại', 'grade', grade_vocab)
        elif 'Khối Lớp' in df.columns:
            df, _ = create_multi_hot(df, 'Khối Lớp', 'grade', grade_vocab)
        if 'Loại tài liệu' in df.columns:
            df, _ = create_multi_hot(df, 'Loại tài liệu', 'material_type', material_type_vocab)
        
        df, subject_pca_cols = apply_pca(df, 'subject', n_components=10)
        df, grade_cols = apply_pca(df, 'grade', n_components=5, skip_if_low_variance=True)
        if 'material_type' in df.columns:
            df, material_type_cols = apply_pca(df, 'material_type', n_components=5)
        
        processed_dataframes.append(df)

    student_data, course_data, tutor_data, material_data, \
    student_course_train, student_course_test, \
    student_tutor_train, student_tutor_test, \
    student_material_train, student_material_test = processed_dataframes

    for df, name in [
        (student_data, 'student_data'),
        (course_data, 'course_data'),
        (tutor_data, 'tutor_data'),
        (material_data, 'material_data'),
        (student_course_train, 'student_course_train'),
        (student_tutor_train, 'student_tutor_train'),
        (student_material_train, 'student_material_train')
    ]:
        subject_cols = [col for col in df.columns if col.startswith('subject_pca_')]
        grade_cols = [col for col in df.columns if col.startswith('grade_pca_') or col.startswith('grade_')]
        if len(subject_cols) != 10 or len(grade_cols) < 5:
            logger.warning(f"Inconsistent columns in {name}: expected 10 subject_pca, at least 5 grade-related")

    course_minority_ids = student_course_train['ID Trung Tâm'].value_counts()[student_course_train['ID Trung Tâm'].value_counts() < 5].index
    upsampled_dfs = [student_course_train]
    for id_ in course_minority_ids:
        minority_df = student_course_train[student_course_train['ID Trung Tâm'] == id_]
        upsampled = resample(minority_df, replace=True, n_samples=5, random_state=42)
        upsampled_dfs.append(upsampled)
    student_course_train = pd.concat(upsampled_dfs)
    if len(student_course_train) > 1000:
        student_course_train = student_course_train.sample(n=1000, random_state=42)

    minority_ids = student_tutor_train['ID Gia Sư'].value_counts()[student_tutor_train['ID Gia Sư'].value_counts() < 5].index
    upsampled_dfs = [student_tutor_train]
    for id_ in minority_ids:
        minority_df = student_tutor_train[student_tutor_train['ID Gia Sư'] == id_]
        upsampled = resample(minority_df, replace=True, n_samples=5, random_state=42)
        upsampled_dfs.append(upsampled)
    student_tutor_train = pd.concat(upsampled_dfs)
    if len(student_tutor_train) > 1000:
        student_tutor_train = student_tutor_train.sample(n=1000, random_state=42)

    minority_ids = student_material_train['ID Tài Liệu'].value_counts()[student_material_train['ID Tài Liệu'].value_counts() < 5].index
    upsampled_dfs = [student_material_train]
    for id_ in minority_ids:
        minority_df = student_material_train[student_material_train['ID Tài Liệu'] == id_]
        upsampled = resample(minority_df, replace=True, n_samples=5, random_state=42)
        upsampled_dfs.append(upsampled)
    student_material_train = pd.concat(upsampled_dfs)
    if len(student_material_train) > 1000:
        student_material_train = student_material_train.sample(n=1000, random_state=42)
    
    # Preprocess categorical and numeric columns
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
    
    numeric_columns = {
        'Chi phí': 'float32',
        'Thời gian': 'float32',
        'Kinh nghiệm giảng dạy': 'float32',
        'Đánh giá': 'float32'
    }
    
    categorical_columns = [
        'Trường học hiện tại', 'Mục tiêu học', 'Phương pháp học yêu thích',
        'Tên Trung Tâm', 'Phương pháp học', 'Tên gia sư', 'Thời gian dạy học',
        'Tên tài liệu', 'Địa chỉ', 'Khối Lớp', 'Khối Lớp hiện tại',
        'Loại tài liệu', 'Môn học', 'Môn học yêu thích'
    ]
    
    def preprocess_categorical(df):
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('unknown').astype(str).str.strip().str.lower()
                df[col] = df[col].str.encode('utf-8').str.decode('utf-8', errors='ignore')
                df[col] = 'cat_' + df[col]
                df[col] = df[col].apply(lambda x: f'cat_{x}' if x.replace('cat_', '').isdigit() else x)
        return df
    
    def preprocess_numeric(df):
        for col, dtype in numeric_columns.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(dtype)
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val + 1e-10)
        return df
    
    for i, df in enumerate([student_data, course_data, tutor_data, material_data,
                           student_course_train, student_course_test,
                           student_tutor_train, student_tutor_test,
                           student_material_train, student_material_test]):
        df = preprocess_categorical(df)
        df = preprocess_numeric(df)
        processed_dataframes[i] = df

    student_data, course_data, tutor_data, material_data, \
    student_course_train, student_course_test, \
    student_tutor_train, student_tutor_test, \
    student_material_train, student_material_test = processed_dataframes
    
    student_course_train.name = 'student_course_train'
    student_tutor_train.name = 'student_tutor_train'
    student_material_train.name = 'student_material_train'
    student_course_test.name = 'student_course_test'
    student_tutor_test.name = 'student_tutor_test'
    student_material_test.name = 'student_material_test'

    student_course_train_dataset = create_tf_dataset(student_course_train)
    student_tutor_train_dataset = create_tf_dataset(student_tutor_train)
    student_material_train_dataset = create_tf_dataset(student_material_train)
    student_course_test_dataset = create_tf_dataset(student_course_test)
    student_tutor_test_dataset = create_tf_dataset(student_tutor_test)
    student_material_test_dataset = create_tf_dataset(student_material_test)
    
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
        'student_course_test_dataset': student_course_test_dataset,
        'student_tutor_test_dataset': student_tutor_test_dataset,
        'student_material_test_dataset': student_material_test_dataset,
        'subject_vocab': subject_vocab,
        'grade_vocab': grade_vocab,
        'material_type_vocab': material_type_vocab,
        'teaching_time_vocab': teaching_time_vocab
    }

class StudentTower(tf.keras.Model):
    def __init__(self, unique_schools, unique_goals, unique_learning_methods, subject_vocab, grade_vocab):
        super().__init__()
        
        unique_schools = np.array([str(x).encode('utf-8') for x in unique_schools], dtype=np.bytes_)
        unique_goals = np.array([str(x).encode('utf-8') for x in unique_goals], dtype=np.bytes_)
        unique_learning_methods = np.array([str(x).encode('utf-8') for x in unique_learning_methods], dtype=np.bytes_)
        self.subject_vocab = np.array([str(x).encode('utf-8') for x in subject_vocab], dtype=np.bytes_)
        self.grade_vocab = np.array([str(x).encode('utf-8') for x in grade_vocab], dtype=np.bytes_)
        
        self.school_lookup = tf.keras.layers.StringLookup(vocabulary=unique_schools, mask_token=None, output_mode='int')
        self.school_embedding = tf.keras.layers.Embedding(input_dim=len(unique_schools) + 2, output_dim=32)
        
        self.goal_lookup = tf.keras.layers.StringLookup(vocabulary=unique_goals, mask_token=None, output_mode='int')
        self.goal_embedding = tf.keras.layers.Embedding(input_dim=len(unique_goals) + 2, output_dim=32)
        
        self.learning_method_lookup = tf.keras.layers.StringLookup(vocabulary=unique_learning_methods, mask_token=None, output_mode='int')
        self.learning_method_embedding = tf.keras.layers.Embedding(input_dim=len(unique_learning_methods) + 2, output_dim=32)
        
        self.subject_dense = tf.keras.layers.Dense(32, input_shape=(10,))
        self.grade_dense = tf.keras.layers.Dense(32, input_shape=(len(grade_vocab),))
        
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001), input_shape=(32*3 + 32 + 32,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32)
        ])

    def call(self, inputs):
        logger.debug(f"StudentTower inputs: {list(inputs.keys())}")
        
        school_indices = self.school_lookup(inputs['Trường học hiện tại'])
        goal_indices = self.goal_lookup(inputs['Mục tiêu học'])
        learning_method_indices = self.learning_method_lookup(inputs['Phương pháp học yêu thích'])
        
        school_embedding = tf.keras.layers.Dropout(0.3)(self.school_embedding(school_indices))
        goal_embedding = self.goal_embedding(goal_indices)
        learning_method_embedding = self.learning_method_embedding(learning_method_indices)
        
        subject_inputs = [inputs.get(f'subject_pca_{i}', tf.zeros_like(inputs['Mục tiêu học'], dtype=tf.float32)) for i in range(10)]
        subject_multi_hot = tf.stack(subject_inputs, axis=-1)
        subject_embedding = self.subject_dense(subject_multi_hot)
        
        grade_inputs = [inputs.get(f'grade_pca_{i}', inputs.get(f'grade_{self.grade_vocab[i].decode()}', tf.zeros_like(inputs['Mục tiêu học'], dtype=tf.float32))) for i in range(len(self.grade_vocab))]
        grade_multi_hot = tf.stack(grade_inputs, axis=-1)
        grade_embedding = self.grade_dense(grade_multi_hot)
        
        concatenated = tf.concat([school_embedding, goal_embedding, learning_method_embedding, subject_embedding, grade_embedding], axis=-1)
        output = self.dense_layers(concatenated)
        return output

class CourseModel(tf.keras.Model):
    def __init__(self, unique_centers, unique_methods, subject_vocab, grade_vocab):
        super().__init__()
        
        unique_centers = np.array([str(x).encode('utf-8') for x in unique_centers], dtype=np.bytes_)
        unique_methods = np.array([str(x).encode('utf-8') for x in unique_methods], dtype=np.bytes_)
        self.subject_vocab = np.array([str(x).encode('utf-8') for x in subject_vocab], dtype=np.bytes_)
        self.grade_vocab = np.array([str(x).encode('utf-8') for x in grade_vocab], dtype=np.bytes_)
        
        self.center_lookup = tf.keras.layers.StringLookup(vocabulary=unique_centers, mask_token=None, output_mode='int')
        self.center_embedding = tf.keras.layers.Embedding(input_dim=len(unique_centers) + 2, output_dim=32)
        
        self.method_lookup = tf.keras.layers.StringLookup(vocabulary=unique_methods, mask_token=None, output_mode='int')
        self.method_embedding = tf.keras.layers.Embedding(input_dim=len(unique_methods) + 2, output_dim=32)
        
        self.subject_dense = tf.keras.layers.Dense(32, input_shape=(10,))
        self.grade_dense = tf.keras.layers.Dense(32, input_shape=(len(grade_vocab),))
        
        self.cost_dense = tf.keras.layers.Dense(32)
        self.time_dense = tf.keras.layers.Dense(32)
        
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005), input_shape=(32*2 + 32 + 32 + 32 + 32,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32)
        ])
        
        self.candidate_embeddings = None

    def call(self, inputs):
        cost = tf.cast(inputs.get('Chi phí', tf.zeros_like(inputs['Tên Trung Tâm'], dtype=tf.float32)), tf.float32)
        time = tf.cast(inputs.get('Thời gian', tf.zeros_like(inputs['Tên Trung Tâm'], dtype=tf.float32)), tf.float32)
        
        cost_2d = tf.expand_dims(cost, axis=-1)
        time_2d = tf.expand_dims(time, axis=-1)
        
        center_indices = self.center_lookup(inputs['Tên Trung Tâm'])
        method_indices = self.method_lookup(inputs['Phương pháp học'])
        
        center_embedding = self.center_embedding(center_indices)
        method_embedding = self.method_embedding(method_indices)
        
        subject_inputs = [inputs.get(f'subject_pca_{i}', tf.zeros_like(inputs['Tên Trung Tâm'], dtype=tf.float32)) for i in range(10)]
        subject_multi_hot = tf.stack(subject_inputs, axis=-1)
        subject_embedding = self.subject_dense(subject_multi_hot)
        
        grade_inputs = [inputs.get(f'grade_pca_{i}', inputs.get(f'grade_{self.grade_vocab[i].decode()}', tf.zeros_like(inputs['Tên Trung Tâm'], dtype=tf.float32))) for i in range(len(self.grade_vocab))]
        grade_multi_hot = tf.stack(grade_inputs, axis=-1)
        grade_embedding = self.grade_dense(grade_multi_hot)
        
        cost_embedding = self.cost_dense(cost_2d)
        time_embedding = self.time_dense(time_2d)
        
        concatenated = tf.concat([center_embedding, method_embedding, subject_embedding, grade_embedding, cost_embedding, time_embedding], axis=-1)
        output = self.dense_layers(concatenated)
        return output

class TutorModel(tf.keras.Model):
    def __init__(self, unique_tutors, unique_teaching_times, subject_vocab, grade_vocab):
        super().__init__()
        
        unique_tutors = np.array([str(x).encode('utf-8') for x in unique_tutors], dtype=np.bytes_)
        unique_teaching_times = np.array([str(x).encode('utf-8') for x in unique_teaching_times], dtype=np.bytes_)
        self.subject_vocab = np.array([str(x).encode('utf-8') for x in subject_vocab], dtype=np.bytes_)
        self.grade_vocab = np.array([str(x).encode('utf-8') for x in grade_vocab], dtype=np.bytes_)
        
        self.tutor_lookup = tf.keras.layers.StringLookup(vocabulary=unique_tutors, mask_token=None, output_mode='int')
        self.tutor_embedding = tf.keras.layers.Embedding(input_dim=len(unique_tutors) + 2, output_dim=32)
        
        self.teaching_time_lookup = tf.keras.layers.StringLookup(vocabulary=unique_teaching_times, mask_token=None, output_mode='int')
        self.teaching_time_embedding = tf.keras.layers.Embedding(input_dim=len(unique_teaching_times) + 2, output_dim=32)
        
        self.subject_dense = tf.keras.layers.Dense(32, input_shape=(10,))
        self.grade_dense = tf.keras.layers.Dense(32, input_shape=(len(grade_vocab),))
        
        self.experience_dense = tf.keras.layers.Dense(32)
        
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001), input_shape=(32*2 + 32 + 32 + 32,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32)
        ])
        
        self.candidate_embeddings = None

    def call(self, inputs):
        experience = tf.cast(inputs.get('Kinh nghiệm giảng dạy', tf.zeros_like(inputs['Tên gia sư'], dtype=tf.float32)), tf.float32)
        experience_2d = tf.expand_dims(experience, axis=-1)
        
        tutor_indices = self.tutor_lookup(inputs['Tên gia sư'])
        tutor_embedding = self.tutor_embedding(tutor_indices)
        
        teaching_time_indices = self.teaching_time_lookup(inputs['Thời gian dạy học'])
        teaching_time_embedding = self.teaching_time_embedding(teaching_time_indices)
        
        subject_inputs = [inputs.get(f'subject_pca_{i}', tf.zeros_like(inputs['Tên gia sư'], dtype=tf.float32)) for i in range(10)]
        subject_multi_hot = tf.stack(subject_inputs, axis=-1)
        subject_embedding = self.subject_dense(subject_multi_hot)
        
        grade_inputs = [inputs.get(f'grade_pca_{i}', inputs.get(f'grade_{self.grade_vocab[i].decode()}', tf.zeros_like(inputs['Tên gia sư'], dtype=tf.float32))) for i in range(len(self.grade_vocab))]
        grade_multi_hot = tf.stack(grade_inputs, axis=-1)
        grade_embedding = self.grade_dense(grade_multi_hot)
        
        experience_embedding = self.experience_dense(experience_2d)
        
        concatenated = tf.concat([tutor_embedding, teaching_time_embedding, subject_embedding, grade_embedding, experience_embedding], axis=-1)
        output = self.dense_layers(concatenated)
        return output

class MaterialModel(tf.keras.Model):
    def __init__(self, unique_materials, subject_vocab, grade_vocab, material_type_vocab):
        super().__init__()
        
        unique_materials = np.array([str(x).encode('utf-8') for x in unique_materials], dtype=np.bytes_)
        self.subject_vocab = np.array([str(x).encode('utf-8') for x in subject_vocab], dtype=np.bytes_)
        self.grade_vocab = np.array([str(x).encode('utf-8') for x in grade_vocab], dtype=np.bytes_)
        self.material_type_vocab = np.array([str(x).encode('utf-8') for x in material_type_vocab], dtype=np.bytes_)
        
        self.material_lookup = tf.keras.layers.StringLookup(vocabulary=unique_materials, mask_token=None, output_mode='int')
        self.material_embedding = tf.keras.layers.Embedding(input_dim=len(unique_materials) + 2, output_dim=32)
        
        self.subject_dense = tf.keras.layers.Dense(32, input_shape=(10,))
        self.grade_dense = tf.keras.layers.Dense(32, input_shape=(len(grade_vocab),))
        self.type_dense = tf.keras.layers.Dense(32, input_shape=(len(material_type_vocab),))
        
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001), input_shape=(32 + 32 + 32 + 32,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32)
        ])
        
        self.candidate_embeddings = None

    def call(self, inputs):
        material_indices = self.material_lookup(inputs['Tên tài liệu'])
        material_embedding = self.material_embedding(material_indices)
        
        subject_inputs = [inputs.get(f'subject_pca_{i}', tf.zeros_like(inputs['Tên tài liệu'], dtype=tf.float32)) for i in range(10)]
        subject_multi_hot = tf.stack(subject_inputs, axis=-1)
        subject_embedding = self.subject_dense(subject_multi_hot)
        
        grade_inputs = [inputs.get(f'grade_pca_{i}', inputs.get(f'grade_{self.grade_vocab[i].decode()}', tf.zeros_like(inputs['Tên tài liệu'], dtype=tf.float32))) for i in range(len(self.grade_vocab))]
        grade_multi_hot = tf.stack(grade_inputs, axis=-1)
        grade_embedding = self.grade_dense(grade_multi_hot)
        
        type_inputs = [inputs.get(f'material_type_pca_{i}', inputs.get(f'material_type_{self.material_type_vocab[i].decode()}', tf.zeros_like(inputs['Tên tài liệu'], dtype=tf.float32))) for i in range(len(self.material_type_vocab))]
        type_multi_hot = tf.stack(type_inputs, axis=-1)
        type_embedding = self.type_dense(type_multi_hot)
        
        concatenated = tf.concat([material_embedding, subject_embedding, grade_embedding, type_embedding], axis=-1)
        output = self.dense_layers(concatenated)
        return output

class RecommendationModel(tfrs.Model):
    def __init__(self, student_features, course_features, tutor_features, material_features,
                 student_course_train, student_tutor_train, student_material_train,
                 subject_vocab, grade_vocab, material_type_vocab, teaching_time_vocab, bruteforce_data_path=None):
        super().__init__()
        
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
        unique_teaching_times = teaching_time_vocab
        unique_materials = [str(x) for x in material_features['Tên tài liệu'].unique()]
        
        self.student_model = StudentTower(unique_schools, unique_goals, unique_learning_methods, subject_vocab, grade_vocab)
        self.course_model = CourseModel(unique_centers, unique_course_methods, subject_vocab, grade_vocab)
        self.tutor_model = TutorModel(unique_tutors, unique_teaching_times, subject_vocab, grade_vocab)
        self.material_model = MaterialModel(unique_materials, subject_vocab, grade_vocab, material_type_vocab)
        
        self.student_course_train = create_tf_dataset(student_course_train).batch(32).cache().prefetch(tf.data.AUTOTUNE)
        self.student_tutor_train = create_tf_dataset(student_tutor_train).batch(32).cache().prefetch(tf.data.AUTOTUNE)
        self.student_material_train = create_tf_dataset(student_material_train).batch(32).cache().prefetch(tf.data.AUTOTUNE)
        
        self.course_features = course_features
        self.tutor_features = tutor_features
        self.material_features = material_features
        
        course_columns = [
            'ID Trung Tâm', 'Tên Trung Tâm', 'Phương pháp học', 'Thời gian', 'Chi phí', 'Địa chỉ', 'Đánh giá'
        ] + [f'subject_pca_{i}' for i in range(10)] + [f'grade_{v}' for v in grade_vocab]
        available_course_columns = [col for col in course_columns if col in course_features.columns]
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
        self.course_model.candidate_embeddings = {
            'embeddings': tf.concat(course_embeddings, axis=0),
            'identifiers': tf.concat(course_ids, axis=0)
        }
        
        tutor_columns = [
            'ID Gia Sư', 'Tên gia sư', 'Thời gian dạy học', 'Kinh nghiệm giảng dạy'
        ] + [f'subject_pca_{i}' for i in range(10)] + [f'grade_{v}' for v in grade_vocab]
        available_tutor_columns = [col for col in tutor_columns if col in tutor_features.columns]
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
        self.tutor_model.candidate_embeddings = {
            'embeddings': tf.concat(tutor_embeddings, axis=0),
            'identifiers': tf.concat(tutor_ids, axis=0)
        }
        
        material_columns = [
            'ID Tài Liệu', 'Tên tài liệu'
        ] + [f'subject_pca_{i}' for i in range(10)] + [f'grade_{v}' for v in grade_vocab] + [f'material_type_{v}' for v in material_type_vocab]
        available_material_columns = [col for col in material_columns if col in material_features.columns]
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
        self.material_model.candidate_embeddings = {
            'embeddings': tf.concat(material_embeddings, axis=0),
            'identifiers': tf.concat(material_ids, axis=0)
        }
        
        self.course_task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=tf.data.Dataset.from_tensor_slices(
                    self.course_model.candidate_embeddings['embeddings']
                ).batch(32)
            )
        )
        self.tutor_task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=tf.data.Dataset.from_tensor_slices(
                    self.tutor_model.candidate_embeddings['embeddings']
                ).batch(32)
            )
        )
        self.material_task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=tf.data.Dataset.from_tensor_slices(
                    self.material_model.candidate_embeddings['embeddings']
                ).batch(32)
            )
        )
        
        self.bruteforce = None
        if self.course_model.candidate_embeddings and self.tutor_model.candidate_embeddings and self.material_model.candidate_embeddings:
            all_embeddings = tf.concat([
                self.course_model.candidate_embeddings['embeddings'],
                self.tutor_model.candidate_embeddings['embeddings'],
                self.material_model.candidate_embeddings['embeddings']
            ], axis=0)
            all_identifiers = tf.concat([
                self.course_model.candidate_embeddings['identifiers'],
                self.tutor_model.candidate_embeddings['identifiers'],
                self.material_model.candidate_embeddings['identifiers']
            ], axis=0)
            self.bruteforce = tfrs.layers.factorized_top_k.BruteForce(k=10)
            self.bruteforce.index(all_embeddings, identifiers=all_identifiers)

    def call(self, inputs, training=False):
        """Define the forward pass for serialization."""
        student_embeddings = self.student_model(inputs)
        
        outputs = {}
        if 'ID Trung Tâm' in inputs:
            course_id_to_index = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    keys=tf.convert_to_tensor(self.course_features['ID Trung Tâm'].values, dtype=tf.string),
                    values=tf.range(tf.shape(self.course_features['ID Trung Tâm'].values)[0], dtype=tf.int32)
                ),
                default_value=0
            )
            course_indices = course_id_to_index.lookup(inputs['ID Trung Tâm'])
            course_batch = {k: tf.gather(v, course_indices) for k, v in self.course_features.items()}
            course_embeddings = self.course_model(course_batch)
            outputs['course'] = course_embeddings
        
        if 'ID Gia Sư' in inputs:
            tutor_id_to_index = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    keys=tf.convert_to_tensor(self.tutor_features['ID Gia Sư'].values, dtype=tf.string),
                    values=tf.range(tf.shape(self.tutor_features['ID Gia Sư'].values)[0], dtype=tf.int32)
                ),
                default_value=0
            )
            tutor_indices = tutor_id_to_index.lookup(inputs['ID Gia Sư'])
            tutor_batch = {k: tf.gather(v, tutor_indices) for k, v in self.tutor_features.items()}
            tutor_embeddings = self.tutor_model(tutor_batch)
            outputs['tutor'] = tutor_embeddings
        
        if 'ID Tài Liệu' in inputs:
            material_id_to_index = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    keys=tf.convert_to_tensor(self.material_features['ID Tài Liệu'].values, dtype=tf.string),
                    values=tf.range(tf.shape(self.material_features['ID Tài Liệu'].values)[0], dtype=tf.int32)
                ),
                default_value=0
            )
            material_indices = material_id_to_index.lookup(inputs['ID Tài Liệu'])
            material_batch = {k: tf.gather(v, material_indices) for k, v in self.material_features.items()}
            material_embeddings = self.material_model(material_batch)
            outputs['material'] = material_embeddings
        
        return outputs

    @tf.function
    def compute_loss(self, inputs, training=False):
        total_loss = 0.0
        batch_count = 0
        
        outputs = self.call(inputs, training=training)
        
        if 'course' in outputs:
            student_embeddings = self.student_model(inputs)
            course_embeddings = outputs['course']
            loss = self.course_task(
                query_embeddings=student_embeddings,
                candidate_embeddings=course_embeddings,
                compute_metrics=not training
            )
            total_loss += loss
            batch_count += 1
        
        if 'tutor' in outputs:
            student_embeddings = self.student_model(inputs)
            tutor_embeddings = outputs['tutor']
            loss = self.tutor_task(
                query_embeddings=student_embeddings,
                candidate_embeddings=tutor_embeddings,
                compute_metrics=not training
            )
            total_loss += loss
            batch_count += 1
        
        if 'material' in outputs:
            student_embeddings = self.student_model(inputs)
            material_embeddings = outputs['material']
            loss = self.material_task(
                query_embeddings=student_embeddings,
                candidate_embeddings=material_embeddings,
                compute_metrics=not training
            )
            total_loss += loss
            batch_count += 1
        
        return total_loss / tf.cast(batch_count, tf.float32) if batch_count > 0 else total_loss

    def save_bruteforce_data(self, path):
        if self.bruteforce is None:
            logger.warning("BruteForce layer not initialized, skipping save.")
            return
        embeddings = tf.concat([
            self.course_model.candidate_embeddings['embeddings'],
            self.tutor_model.candidate_embeddings['embeddings'],
            self.material_model.candidate_embeddings['embeddings']
        ], axis=0)
        identifiers = tf.concat([
            self.course_model.candidate_embeddings['identifiers'],
            self.tutor_model.candidate_embeddings['identifiers'],
            self.material_model.candidate_embeddings['identifiers']
        ], axis=0)
        try:
            np.savez(path, embeddings=embeddings.numpy(), identifiers=identifiers.numpy())
            logger.info(f"BruteForce data saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save BruteForce data: {str(e)}")

    def load_bruteforce_data(self, path):
        try:
            data = np.load(path)
            embeddings = tf.convert_to_tensor(data['embeddings'], dtype=tf.float32)
            identifiers = tf.convert_to_tensor(data['identifiers'], dtype=tf.string)
            self.bruteforce = tfrs.layers.factorized_top_k.BruteForce(k=10)
            self.bruteforce.index(embeddings, identifiers=identifiers)
            logger.info(f"BruteForce data loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load BruteForce data from {path}: {str(e)}")
