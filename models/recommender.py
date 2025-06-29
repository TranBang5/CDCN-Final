import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.decomposition import PCA
import logging
import re
import os
import uuid

# Suppress TensorFlow warnings for cleaner logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mapping of original column names to ASCII-compliant names
name_mapping = {
    'Trường học hiện tại': 'truong_hoc_hien_tai',
    'Mục tiêu học': 'muc_tieu_hoc',
    'Phương pháp học yêu thích': 'phuong_phap_hoc_yeu_thich',
    'Tên Trung Tâm': 'ten_trung_tam',
    'Phương pháp học': 'phuong_phap_hoc',
    'Tên gia sư': 'ten_gia_su',
    'Thời gian dạy học': 'thoi_gian_day_hoc',
    'Tên tài liệu': 'ten_tai_lieu',
    'Địa chỉ': 'dia_chi',
    'Khối Lớp': 'khoi_lop',
    'Khối Lớp hiện tại': 'khoi_lop_hien_tai',
    'Loại tài liệu': 'loai_tai_lieu',
    'Môn học': 'mon_hoc',
    'Môn học yêu thích': 'mon_hoc_yeu_thich',
    'ID Học Sinh': 'id_hoc_sinh',
    'ID Trung Tâm': 'id_trung_tam',
    'ID Gia Sư': 'id_gia_su',
    'ID Tài Liệu': 'id_tai_lieu',
    'Chi phí': 'chi_phi',
    'Thời gian': 'thoi_gian',
    'Kinh nghiệm giảng dạy': 'kinh_nghiem_giang_day',
    'Đánh giá': 'danh_gia',
    'Tên': 'ten',
    'material_type_Giấy': 'material_type_giay',
    'material_type_Điện tử': 'material_type_dien_tu'
}

# Function to sanitize strings to ASCII-compliant names
def sanitize_name(name):
    if not isinstance(name, str):
        name = str(name)
    # Replace Vietnamese diacritics and special characters
    replacements = {
        'àáảãạăằắẳẵặâầấẩẫậ': 'a',
        'èéẻẽẹêềếểễệ': 'e',
        'ìíỉĩị': 'i',
        'òóỏõọôồốổỗộơờớởỡợ': 'o',
        'ùúủũụưừứửữự': 'u',
        'ỳýỷỹỵ': 'y',
        'đ': 'd',
        ' ': '_',
        'Lớp': 'lop',
        'Giấy': 'giay',
        'Điện tử': 'dien_tu'
    }
    name = name.lower()
    for chars, replacement in replacements.items():
        for char in chars:
            name = name.replace(char, replacement)
    # Remove any remaining non-ASCII characters
    name = re.sub(r'[^A-Za-z0-9_]', '_', name)
    # Ensure name starts with a valid character
    if not name[0].isalpha() and name[0] != '_':
        name = 'v_' + name
    return name

def create_tf_dataset(df):
    dtype_dict = {}
    categorical_columns = [
        'truong_hoc_hien_tai', 'muc_tieu_hoc', 'phuong_phap_hoc_yeu_thich',
        'ten_trung_tam', 'phuong_phap_hoc', 'ten_gia_su', 'thoi_gian_day_hoc',
        'ten_tai_lieu', 'dia_chi', 'khoi_lop', 'khoi_lop_hien_tai',
        'loai_tai_lieu', 'mon_hoc', 'mon_hoc_yeu_thich', 'ten'
    ]
    id_columns = ['id_hoc_sinh', 'id_trung_tam', 'id_gia_su', 'id_tai_lieu']
    numeric_columns = {
        'chi_phi': 'float32',
        'thoi_gian': 'float32',
        'kinh_nghiem_giang_day': 'int32',
        'danh_gia': 'float32'
    }
    multi_hot_columns = [col for col in df.columns if col.startswith('subject_') or col.startswith('grade_') or col.startswith('material_type_')]
    
    for col in df.columns:
        if col in categorical_columns or col in id_columns:
            dtype_dict[col] = tf.string
        elif col in numeric_columns or col in multi_hot_columns or col.startswith('subject_pca_') or col.startswith('grade_pca_') or col.startswith('material_type_pca_'):
            dtype_dict[col] = tf.float32
        else:
            dtype_dict[col] = tf.string
            logger.warning(f"Column {col} not explicitly typed, defaulting to string")
    
    # Preprocess the DataFrame
    for col in df.columns:
        if col in categorical_columns or col in id_columns:
            df[col] = df[col].fillna('unknown').astype(str).str.encode('utf-8').str.decode('utf-8', errors='ignore')
            if df[col].str.contains('counter', case=False, na=False).any():
                logger.warning(f"Found 'counter' in column {col}: {df[col][df[col].str.contains('counter', case=False, na=False)].unique()}")
                df[col] = df[col].replace(to_replace=r'.*counter.*', value='unknown', regex=True)
            logger.debug(f"Column {col}: unique values = {df[col].unique()[:5]}")
        elif col in numeric_columns or col in multi_hot_columns or col.startswith('subject_pca_') or col.startswith('grade_pca_') or col.startswith('material_type_pca_'):
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float32)
        else:
            logger.warning(f"Column {col} not preprocessed, defaulting to string")
            df[col] = df[col].fillna('unknown').astype(str).str.encode('utf-8').str.decode('utf-8', errors='ignore')
            if df[col].str.contains('counter', case=False, na=False).any():
                logger.warning(f"Found 'counter' in column {col}: {df[col][df[col].str.contains('counter', case=False, na=False)].unique()}")
                df[col] = df[col].replace(to_replace=r'.*counter.*', value='unknown', regex=True)
    
    data_dict = {col: df[col].values for col in df.columns}
    for col, dtype in dtype_dict.items():
        if col not in data_dict:
            logger.warning(f"Column {col} missing, filling with zeros")
            data_dict[col] = np.zeros(len(df), dtype=np.bytes_ if dtype == tf.string else np.float32)
        elif dtype == tf.string:
            data_dict[col] = np.array([
                x.encode('utf-8') if isinstance(x, str) and x and 'counter' not in x.lower() else b'unknown'
                for x in data_dict[col]
            ], dtype=np.bytes_)
        else:
            data_dict[col] = np.array(data_dict[col], dtype=np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices(data_dict)
    return dataset

class PrecisionAtK(tf.keras.metrics.Metric):
    def __init__(self, k=10, name='precision_at_k', **kwargs):
        super(PrecisionAtK, self).__init__(name=name, **kwargs)
        self.k = k
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.predicted_positives = self.add_weight(name='pp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true: ground-truth item IDs (string tensor)
        # y_pred: top-K predicted item IDs (string tensor, shape [batch_size, k])
        y_true = tf.cast(y_true, tf.string)
        y_pred = tf.cast(y_pred, tf.string)

        # For each sample, check if the ground-truth ID is in the top-K predictions
        true_positives = tf.reduce_sum(tf.cast(tf.reduce_any(tf.equal(
            tf.expand_dims(y_true, axis=-1), y_pred), axis=-1), tf.float32))
        
        # Number of predicted items is k per sample
        predicted_positives = tf.cast(tf.shape(y_pred)[0] * self.k, tf.float32)

        self.true_positives.assign_add(true_positives)
        self.predicted_positives.assign_add(predicted_positives)

    def result(self):
        return self.true_positives / (self.predicted_positives + 1e-10)

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.predicted_positives.assign(0.0)

class RecallAtK(tf.keras.metrics.Metric):
    def __init__(self, k=10, name='recall_at_k', **kwargs):
        super(RecallAtK, self).__init__(name=name, **kwargs)
        self.k = k
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.relevant_items = self.add_weight(name='ri', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true: ground-truth item IDs (string tensor)
        # y_pred: top-K predicted item IDs (string tensor, shape [batch_size, k])
        y_true = tf.cast(y_true, tf.string)
        y_pred = tf.cast(y_pred, tf.string)

        # For each sample, check if the ground-truth ID is in the top-K predictions
        true_positives = tf.reduce_sum(tf.cast(tf.reduce_any(tf.equal(
            tf.expand_dims(y_true, axis=-1), y_pred), axis=-1), tf.float32))
        
        # Number of relevant items is the number of ground-truth items (batch_size)
        relevant_items = tf.cast(tf.shape(y_true)[0], tf.float32)

        self.true_positives.assign_add(true_positives)
        self.relevant_items.assign_add(relevant_items)

    def result(self):
        return self.true_positives / (self.relevant_items + 1e-10)

    def reset_states(self):
        self.true_positives.assign(0.0)
        self.relevant_items.assign(0.0)

def load_and_preprocess_data():
    categorical_dtypes = {name_mapping[old]: str for old in [
        'Trường học hiện tại', 'Khối Lớp hiện tại', 'Mục tiêu học',
        'Môn học yêu thích', 'Phương pháp học yêu thích', 'Tên Trung Tâm',
        'Môn học', 'Khối Lớp', 'Phương pháp học', 'Tên gia sư',
        'Thời gian dạy học', 'Tên tài liệu', 'Loại tài liệu', 'Địa chỉ', 'Tên'
    ] if old in name_mapping}
    
    logger.info("Loading CSV files...")
    student_data = pd.read_csv('data/hoc_sinh.csv', dtype=categorical_dtypes, encoding='utf-8')
    course_data = pd.read_csv('data/trung_tam.csv', dtype=categorical_dtypes, encoding='utf-8')
    tutor_data = pd.read_csv('data/gia_su.csv', dtype=categorical_dtypes, encoding='utf-8')
    material_data = pd.read_csv('data/tai_lieu.csv', dtype=categorical_dtypes, encoding='utf-8')
    
    # Rename columns to ASCII-compliant names
    for df in [student_data, course_data, tutor_data, material_data]:
        df.rename(columns=name_mapping, inplace=True)
    
    # Clean 'thoi_gian_day_hoc' in tutor_data
    if 'thoi_gian_day_hoc' in tutor_data.columns:
        def clean_teaching_time(value):
            if not isinstance(value, str) or not value or 'counter' in value.lower():
                return 'unknown'
            times = [t.strip() for t in value.split(';') if t.strip()]
            valid_times = []
            for t in times:
                if re.match(r'^\d{1,2}h(\s+thu\s+\w+|\s+chu_nhat)?$', t, re.IGNORECASE):
                    valid_times.append(sanitize_name(t))
                else:
                    logger.debug(f"Invalid time format in 'thoi_gian_day_hoc': {t}")
            return ';'.join(valid_times) if valid_times else 'unknown'
        tutor_data['thoi_gian_day_hoc'] = tutor_data['thoi_gian_day_hoc'].apply(clean_teaching_time)
        logger.info(f"Cleaned 'thoi_gian_day_hoc': unique values = {tutor_data['thoi_gian_day_hoc'].unique()[:10]}")
    
    # Log unique values for key columns
    for df, name in [(student_data, 'student_data'), (course_data, 'course_data'), 
                     (tutor_data, 'tutor_data'), (material_data, 'material_data')]:
        for old_col, new_col in [('Môn học yêu thích', 'mon_hoc_yeu_thich'), ('Môn học', 'mon_hoc'), 
                                 ('Khối Lớp hiện tại', 'khoi_lop_hien_tai'), ('Khối Lớp', 'khoi_lop'), 
                                 ('Loại tài liệu', 'loai_tai_lieu'), ('Thời gian dạy học', 'thoi_gian_day_hoc'), ('Tên', 'ten')]:
            if new_col in df.columns:
                unique_vals = df[new_col].unique()
                if any('counter' in str(v).lower() for v in unique_vals):
                    logger.warning(f"Found 'counter' in {name} - {new_col}: {unique_vals}")
                logger.info(f"{name} - {new_col}: unique values = {unique_vals[:10]} (total: {len(unique_vals)})")
    
    # Validate grade columns
    for df, name in [(student_data, 'student_data'), (course_data, 'course_data')]:
        for col in ['khoi_lop_hien_tai', 'khoi_lop']:
            if col in df.columns:
                unique_grades = df[col].unique()
                logger.info(f"{name} - {col}: unique values = {unique_grades}")
                if len(unique_grades) <= 1:
                    logger.warning(f"{name} - {col} has insufficient unique values: {unique_grades}")
    
    if 'Phương pháp dạy' in tutor_data.columns:
        tutor_data = tutor_data.drop(columns=['Phương pháp dạy'])
    
    if 'kinh_nghiem_giang_day' in tutor_data.columns:
        def extract_years(value):
            if pd.isna(value) or not isinstance(value, str):
                return 0.0
            match = re.match(r'(\d+\.?\d*)\s*nam', value.strip(), re.IGNORECASE)
            return float(match.group(1)) if match else 0.0
        tutor_data['kinh_nghiem_giang_day'] = tutor_data['kinh_nghiem_giang_day'].apply(extract_years)
    
    student_course_train = pd.read_csv('data/hoc_sinh_trung_tam_train.csv', dtype=categorical_dtypes, encoding='utf-8')
    student_course_test = pd.read_csv('data/hoc_sinh_trung_tam_test.csv', dtype=categorical_dtypes, encoding='utf-8')
    student_tutor_train = pd.read_csv('data/hoc_sinh_gia_su_train.csv', dtype=categorical_dtypes, encoding='utf-8')
    student_tutor_test = pd.read_csv('data/hoc_sinh_gia_su_test.csv', dtype=categorical_dtypes, encoding='utf-8')
    student_material_train = pd.read_csv('data/hoc_sinh_tai_lieu_train.csv', dtype=categorical_dtypes, encoding='utf-8')
    student_material_test = pd.read_csv('data/hoc_sinh_tai_lieu_test.csv', dtype=categorical_dtypes, encoding='utf-8')
    
    # Rename columns in train/test datasets
    for df in [student_course_train, student_course_test, student_tutor_train, student_tutor_test, 
               student_material_train, student_material_test]:
        df.rename(columns=name_mapping, inplace=True)
    
    # Validate IDs
    logger.info("Validating IDs...")
    course_ids = set(course_data['id_trung_tam'].astype(str))
    tutor_ids = set(tutor_data['id_gia_su'].astype(str))
    material_ids = set(material_data['id_tai_lieu'].astype(str))
    
    for df, name, id_col in [
        (student_course_train, 'student_course_train', 'id_trung_tam'),
        (student_course_test, 'student_course_test', 'id_trung_tam'),
        (student_tutor_train, 'student_tutor_train', 'id_gia_su'),
        (student_tutor_test, 'student_tutor_test', 'id_gia_su'),
        (student_material_train, 'student_material_train', 'id_tai_lieu'),
        (student_material_test, 'student_material_test', 'id_tai_lieu')
    ]:
        valid_ids = course_ids if id_col == 'id_trung_tam' else tutor_ids if id_col == 'id_gia_su' else material_ids
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
                df[col] = 0.0 if col in ['chi_phi', 'thoi_gian', 'kinh_nghiem_giang_day', 'danh_gia'] else 'unknown'
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
            values = df[column].astype(str).str.encode('utf-8').str.decode('utf-8', errors='ignore').str.split(',').explode().str.strip()
            unique_values = values[values != ''].unique()
            logger.info(f"create_multi_hot - {column}: unique values = {unique_values[:10]}")
            vocab = sorted([sanitize_name(v) for v in unique_values if v and v != 'nan' and v != '0.0' and 
                           not v.replace('.', '').isdigit() and v != 'counter' and isinstance(v, str)])
        if not vocab:
            logger.warning(f"Empty vocabulary for {column}, using ['unknown']")
            vocab = ['unknown']
        multi_hot = np.zeros((len(df), len(vocab)), dtype=np.float32)
        for i, row in enumerate(df[column].astype(str).str.encode('utf-8').str.decode('utf-8', errors='ignore')):
            if row == 'nan' or not row or row == '0.0' or row.replace('.', '').isdigit():
                if column == 'loai_tai_lieu':
                    values = ['dien_tu']
                else:
                    continue
            else:
                values = [v.strip() for v in row.split(',') if v.strip()]
            normalized_values = [sanitize_name(v) for v in values if v != 'counter' and isinstance(v, str)]
            for val in normalized_values:
                if val in vocab:
                    multi_hot[i, vocab.index(val)] = 1.0
        multi_hot_df = pd.DataFrame(multi_hot, columns=[f"{prefix}_{v}" for v in vocab], index=df.index)
        return pd.concat([df, multi_hot_df], axis=1), vocab
    
    def consolidate_vocab(df_list, column):
        values = set()
        for df in df_list:
            if column in df.columns:
                exploded = df[column].astype(str).str.strip().str.encode('utf-8').str.decode('utf-8', errors='ignore')
                exploded = exploded.str.split(',').explode().str.strip()
                for v in exploded:
                    logger.debug(f"Processing value for {column}: {v} (type: {type(v)})")
                    if (v and v != 'nan' and v != '0.0' and 
                        not v.replace('.', '').isdigit() and 
                        v.lower() != 'counter' and 
                        isinstance(v, str) and
                        v != 'ba;16h'):
                        values.add(sanitize_name(v))
        vocab = sorted(list(values))
        logger.info(f"Consolidated vocab for {column}: {vocab[:10]} (total: {len(vocab)})")
        if not vocab:
            logger.warning(f"Empty vocabulary for {column}, using default ['unknown']")
            vocab = ['unknown']
        return vocab
    
    subject_vocab = consolidate_vocab([student_data, course_data, tutor_data, material_data, 
                                      student_course_train, student_course_test, 
                                      student_tutor_train, student_tutor_test, 
                                      student_material_train, student_material_test], 
                                     'mon_hoc_yeu_thich')
    grade_vocab = consolidate_vocab([student_data, course_data, tutor_data, material_data, 
                                    student_course_train, student_course_test, 
                                    student_tutor_train, student_tutor_test, 
                                    student_material_train, student_material_test], 
                                   'khoi_lop_hien_tai')
    material_type_vocab = consolidate_vocab([material_data, student_material_train, student_material_test], 
                                           'loai_tai_lieu')
    teaching_time_vocab = consolidate_vocab([tutor_data, student_tutor_train, student_tutor_test], 
                                           'thoi_gian_day_hoc')

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
        if 'mon_hoc_yeu_thich' in df.columns:
            df, _ = create_multi_hot(df, 'mon_hoc_yeu_thich', 'subject', subject_vocab)
        elif 'mon_hoc' in df.columns:
            df, _ = create_multi_hot(df, 'mon_hoc', 'subject', subject_vocab)
        if 'khoi_lop_hien_tai' in df.columns:
            df, _ = create_multi_hot(df, 'khoi_lop_hien_tai', 'grade', grade_vocab)
        elif 'khoi_lop' in df.columns:
            df, _ = create_multi_hot(df, 'khoi_lop', 'grade', grade_vocab)
        if 'loai_tai_lieu' in df.columns:
            df, _ = create_multi_hot(df, 'loai_tai_lieu', 'material_type', material_type_vocab)
        
        df, subject_pca_cols = apply_pca(df, 'subject', n_components=10)
        df, grade_cols = apply_pca(df, 'grade', n_components=5, skip_if_low_variance=True)
        if 'loai_tai_lieu' in df.columns:
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

    course_minority_ids = student_course_train['id_trung_tam'].value_counts()[student_course_train['id_trung_tam'].value_counts() < 5].index
    upsampled_dfs = [student_course_train]
    for id_ in course_minority_ids:
        minority_df = student_course_train[student_course_train['id_trung_tam'] == id_]
        upsampled = resample(minority_df, replace=True, n_samples=5, random_state=42)
        upsampled_dfs.append(upsampled)
    student_course_train = pd.concat(upsampled_dfs)

    minority_ids = student_tutor_train['id_gia_su'].value_counts()[student_tutor_train['id_gia_su'].value_counts() < 5].index
    upsampled_dfs = [student_tutor_train]
    for id_ in minority_ids:
        minority_df = student_tutor_train[student_tutor_train['id_gia_su'] == id_]
        upsampled = resample(minority_df, replace=True, n_samples=5, random_state=42)
        upsampled_dfs.append(upsampled)
    student_tutor_train = pd.concat(upsampled_dfs)

    minority_ids = student_material_train['id_tai_lieu'].value_counts()[student_material_train['id_tai_lieu'].value_counts() < 5].index
    upsampled_dfs = [student_material_train]
    for id_ in minority_ids:
        minority_df = student_material_train[student_material_train['id_tai_lieu'] == id_]
        upsampled = resample(minority_df, replace=True, n_samples=5, random_state=42)
        upsampled_dfs.append(upsampled)
    student_material_train = pd.concat(upsampled_dfs)
    
    # Preprocess categorical and numeric columns
    valid_goals = set(student_data['muc_tieu_hoc'].astype(str))
    for df in [student_course_train, student_tutor_train, student_material_train]:
        df['muc_tieu_hoc'] = df['muc_tieu_hoc'].apply(lambda x: str(x) if str(x) in valid_goals else 'cat_unknown')
    valid_subjects = set(student_data['mon_hoc_yeu_thich'].astype(str))
    for df in [student_course_train, student_tutor_train, student_material_train]:
        df['mon_hoc_yeu_thich'] = df['mon_hoc_yeu_thich'].apply(lambda x: str(x) if str(x) in valid_subjects else 'cat_unknown')

    id_columns = ['id_hoc_sinh', 'id_trung_tam', 'id_gia_su', 'id_tai_lieu']
    for df in [student_data, course_data, tutor_data, material_data,
               student_course_train, student_course_test,
               student_tutor_train, student_tutor_test,
               student_material_train, student_material_test]:
        for col in id_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                df[col] = df[col].astype(str)
    
    numeric_columns = {
        'chi_phi': 'float32',
        'thoi_gian': 'float32',
        'kinh_nghiem_giang_day': 'float32',
        'danh_gia': 'float32'
    }
    
    categorical_columns = [
        'truong_hoc_hien_tai', 'muc_tieu_hoc', 'phuong_phap_hoc_yeu_thich',
        'ten_trung_tam', 'phuong_phap_hoc', 'ten_gia_su', 'thoi_gian_day_hoc',
        'ten_tai_lieu', 'dia_chi', 'khoi_lop', 'khoi_lop_hien_tai',
        'loai_tai_lieu', 'mon_hoc', 'mon_hoc_yeu_thich', 'ten'
    ]
    
    def preprocess_categorical(df):
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('unknown').astype(str).str.strip()
                df[col] = df[col].apply(sanitize_name)
                df[col] = 'cat_' + df[col]
                df[col] = df[col].apply(lambda x: f'cat_{x}' if x.replace('cat_', '').isdigit() else x)
                if df[col].str.contains('counter', case=False, na=False).any():
                    logger.warning(f"Found 'counter' in {col} after preprocessing: {df[col][df[col].str.contains('counter', case=False, na=False)].unique()}")
                    df[col] = df[col].replace(to_replace=r'.*counter.*', value='cat_unknown', regex=True)
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
        
        unique_schools = [sanitize_name(x) for x in unique_schools if isinstance(x, str) and x and x.lower() != 'counter']
        unique_goals = [sanitize_name(x) for x in unique_goals if isinstance(x, str) and x and x.lower() != 'counter']
        unique_learning_methods = [sanitize_name(x) for x in unique_learning_methods if isinstance(x, str) and x and x.lower() != 'counter']
        subject_vocab = [sanitize_name(x) for x in subject_vocab if isinstance(x, str) and x and x.lower() != 'counter']
        grade_vocab = [sanitize_name(x) for x in grade_vocab if isinstance(x, str) and x and x.lower() != 'counter']
        
        if not subject_vocab:
            logger.warning("Empty subject_vocab, using ['unknown']")
            subject_vocab = ['unknown']
        if not grade_vocab:
            logger.warning("Empty grade_vocab, using ['unknown']")
            grade_vocab = ['unknown']
        
        logger.info(f"StudentTower vocab sizes: schools={len(unique_schools)}, goals={len(unique_goals)}, "
                    f"methods={len(unique_learning_methods)}, subjects={len(subject_vocab)}, grades={len(grade_vocab)}")
        logger.debug(f"subject_vocab: {subject_vocab[:10]}")
        logger.debug(f"grade_vocab: {grade_vocab[:10]}")
        
        self.subject_vocab = np.array(subject_vocab, dtype=np.bytes_)
        self.grade_vocab = np.array(grade_vocab, dtype=np.bytes_)
        
        self._school_lookup = tf.keras.layers.StringLookup(
            vocabulary=np.array(unique_schools, dtype=np.bytes_), 
            mask_token=None, output_mode='int', name='school_lookup')
        self._school_embedding = tf.keras.layers.Embedding(input_dim=len(unique_schools) + 2, output_dim=64, name='school_embedding')
        
        self._goal_lookup = tf.keras.layers.StringLookup(
            vocabulary=np.array(unique_goals, dtype=np.bytes_), 
            mask_token=None, output_mode='int', name='goal_lookup')
        self._goal_embedding = tf.keras.layers.Embedding(input_dim=len(unique_goals) + 2, output_dim=64, name='goal_embedding')
        
        self._learning_method_lookup = tf.keras.layers.StringLookup(
            vocabulary=np.array(unique_learning_methods, dtype=np.bytes_), 
            mask_token=None, output_mode='int', name='learning_method_lookup')
        self._learning_method_embedding = tf.keras.layers.Embedding(input_dim=len(unique_learning_methods) + 2, output_dim=64, name='learning_method_embedding')
        
        self._subject_dense = tf.keras.layers.Dense(32, name='subject_dense')
        self._grade_dense = tf.keras.layers.Dense(32, name='grade_dense')
        
        self._dense_layers = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(64*3 + 32 + 32,), name='student_dense_input'),
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01), name='dense_1'),
            tf.keras.layers.Dropout(0.2, name='dropout_1'),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01), name='dense_2'),
            tf.keras.layers.Dropout(0.2, name='dropout_2'),
            tf.keras.layers.Dense(32, name='dense_output')
        ], name='student_dense_layers')
    
    def call(self, inputs):
        logger.debug(f"StudentTower inputs: {list(inputs.keys())}")
        
        school_indices = self._school_lookup(inputs['truong_hoc_hien_tai'])
        goal_indices = self._goal_lookup(inputs['muc_tieu_hoc'])
        learning_method_indices = self._learning_method_lookup(inputs['phuong_phap_hoc_yeu_thich'])
        
        school_embedding = tf.keras.layers.Dropout(0.3, name='school_dropout')(self._school_embedding(school_indices))
        goal_embedding = self._goal_embedding(goal_indices)
        learning_method_embedding = self._learning_method_embedding(learning_method_indices)
        
        subject_inputs = [inputs.get(f'subject_pca_{i}', tf.zeros_like(inputs['muc_tieu_hoc'], dtype=tf.float32)) for i in range(10)]
        subject_multi_hot = tf.stack(subject_inputs, axis=-1)
        subject_embedding = self._subject_dense(subject_multi_hot)
        
        grade_inputs = [inputs.get(f'grade_pca_{i}', tf.zeros_like(inputs['muc_tieu_hoc'], dtype=tf.float32)) for i in range(5)]
        grade_multi_hot = tf.stack(grade_inputs, axis=-1)
        grade_embedding = self._grade_dense(grade_multi_hot)
        
        concatenated = tf.concat([school_embedding, goal_embedding, learning_method_embedding, subject_embedding, grade_embedding], axis=-1)
        output = self._dense_layers(concatenated)
        return output
    
    def get_config(self):
        config = {
            'unique_schools': list(self._school_lookup.get_vocabulary()),
            'unique_goals': list(self._goal_lookup.get_vocabulary()),
            'unique_learning_methods': list(self._learning_method_lookup.get_vocabulary()),
            'subject_vocab': list(self.subject_vocab.astype(str)),
            'grade_vocab': list(self.grade_vocab.astype(str))
        }
        logger.debug(f"StudentTower get_config: {config}")
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(
            unique_schools=config['unique_schools'],
            unique_goals=config['unique_goals'],
            unique_learning_methods=config['unique_learning_methods'],
            subject_vocab=config['subject_vocab'],
            grade_vocab=config['grade_vocab']
        )

class CourseModel(tf.keras.Model):
    def __init__(self, unique_centers, unique_methods, subject_vocab, grade_vocab):
        super().__init__()
        
        unique_centers = [sanitize_name(x) for x in unique_centers if isinstance(x, str) and x and x.lower() != 'counter']
        unique_methods = [sanitize_name(x) for x in unique_methods if isinstance(x, str) and x and x.lower() != 'counter']
        subject_vocab = [sanitize_name(x) for x in subject_vocab if isinstance(x, str) and x and x.lower() != 'counter']
        grade_vocab = [sanitize_name(x) for x in grade_vocab if isinstance(x, str) and x and x.lower() != 'counter']
        
        if not subject_vocab:
            logger.warning("Empty subject_vocab, using ['unknown']")
            subject_vocab = ['unknown']
        if not grade_vocab:
            logger.warning("Empty grade_vocab, using ['unknown']")
            grade_vocab = ['unknown']
        
        logger.info(f"CourseModel vocab sizes: centers={len(unique_centers)}, methods={len(unique_methods)}, "
                    f"subjects={len(subject_vocab)}, grades={len(grade_vocab)}")
        logger.debug(f"subject_vocab: {subject_vocab[:10]}")
        logger.debug(f"grade_vocab: {grade_vocab[:10]}")
        
        self.subject_vocab = np.array(subject_vocab, dtype=np.bytes_)
        self.grade_vocab = np.array(grade_vocab, dtype=np.bytes_)
        
        self._center_lookup = tf.keras.layers.StringLookup(
            vocabulary=np.array(unique_centers, dtype=np.bytes_), 
            mask_token=None, output_mode='int', name='center_lookup')
        self._center_embedding = tf.keras.layers.Embedding(input_dim=len(unique_centers) + 2, output_dim=64, name='center_embedding')
        
        self._method_lookup = tf.keras.layers.StringLookup(
            vocabulary=np.array(unique_methods, dtype=np.bytes_), 
            mask_token=None, output_mode='int', name='method_lookup')
        self._method_embedding = tf.keras.layers.Embedding(input_dim=len(unique_methods) + 2, output_dim=64, name='method_embedding')
        
        self._subject_dense = tf.keras.layers.Dense(32, name='subject_dense')
        self._grade_dense = tf.keras.layers.Dense(32, name='grade_dense')
        
        self._cost_dense = tf.keras.layers.Dense(32, name='cost_dense')
        self._time_dense = tf.keras.layers.Dense(32, name='time_dense')
        
        self._dense_layers = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(64*2 + 32 + 32 + 32 + 32,), name='course_dense_input'),
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01), name='dense_1'),
            tf.keras.layers.Dropout(0.2, name='dropout_1'),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01), name='dense_2'),
            tf.keras.layers.Dropout(0.2, name='dropout_2'),
            tf.keras.layers.Dense(32, name='dense_output')
        ], name='course_dense_layers')
        
    def call(self, inputs):
        cost = tf.cast(inputs.get('chi_phi', tf.zeros_like(inputs['ten_trung_tam'], dtype=tf.float32)), tf.float32)
        time = tf.cast(inputs.get('thoi_gian', tf.zeros_like(inputs['ten_trung_tam'], dtype=tf.float32)), tf.float32)
        
        cost_2d = tf.expand_dims(cost, axis=-1)
        time_2d = tf.expand_dims(time, axis=-1)
        
        center_indices = self._center_lookup(inputs['ten_trung_tam'])
        method_indices = self._method_lookup(inputs['phuong_phap_hoc'])
        
        center_embedding = self._center_embedding(center_indices)
        method_embedding = self._method_embedding(method_indices)
        
        subject_inputs = [inputs.get(f'subject_pca_{i}', tf.zeros_like(inputs['ten_trung_tam'], dtype=tf.float32)) for i in range(10)]
        subject_multi_hot = tf.stack(subject_inputs, axis=-1)
        subject_embedding = self._subject_dense(subject_multi_hot)
        
        grade_inputs = [inputs.get(f'grade_pca_{i}', tf.zeros_like(inputs['ten_trung_tam'], dtype=tf.float32)) for i in range(5)]
        grade_multi_hot = tf.stack(grade_inputs, axis=-1)
        grade_embedding = self._grade_dense(grade_multi_hot)
        
        cost_embedding = self._cost_dense(cost_2d)
        time_embedding = self._time_dense(time_2d)
        
        concatenated = tf.concat([center_embedding, method_embedding, subject_embedding, grade_embedding, cost_embedding, time_embedding], axis=-1)
        output = self._dense_layers(concatenated)
        return output
    
    def get_config(self):
        config = {
            'unique_centers': list(self._center_lookup.get_vocabulary()),
            'unique_methods': list(self._method_lookup.get_vocabulary()),
            'subject_vocab': list(self.subject_vocab.astype(str)),
            'grade_vocab': list(self.grade_vocab.astype(str))
        }
        logger.debug(f"CourseModel get_config: {config}")
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(
            unique_centers=config['unique_centers'],
            unique_methods=config['unique_methods'],
            subject_vocab=config['subject_vocab'],
            grade_vocab=config['grade_vocab']
        )

class TutorModel(tf.keras.Model):
    def __init__(self, unique_tutors, unique_teaching_times, subject_vocab, grade_vocab):
        super().__init__()
        
        unique_tutors = [sanitize_name(x) for x in unique_tutors if isinstance(x, str) and x and x.lower() != 'counter']
        unique_teaching_times = [sanitize_name(x) for x in unique_teaching_times if isinstance(x, str) and x and x.lower() != 'counter']
        subject_vocab = [sanitize_name(x) for x in subject_vocab if isinstance(x, str) and x and x.lower() != 'counter']
        grade_vocab = [sanitize_name(x) for x in grade_vocab if isinstance(x, str) and x and x.lower() != 'counter']
        
        if not subject_vocab:
            logger.warning("Empty subject_vocab, using ['unknown']")
            subject_vocab = ['unknown']
        if not grade_vocab:
            logger.warning("Empty grade_vocab, using ['unknown']")
            grade_vocab = ['unknown']
        if not unique_tutors:
            logger.warning("Empty unique_tutors, using ['unknown']")
            unique_tutors = ['unknown']
        if not unique_teaching_times:
            logger.warning("Empty unique_teaching_times, using ['unknown']")
            unique_teaching_times = ['unknown']
        
        logger.info(f"TutorModel vocab sizes: tutors={len(unique_tutors)}, times={len(unique_teaching_times)}, "
                    f"subjects={len(subject_vocab)}, grades={len(grade_vocab)}")
        logger.debug(f"subject_vocab: {subject_vocab[:10]}")
        logger.debug(f"grade_vocab: {grade_vocab[:10]}")
        logger.debug(f"unique_tutors: {unique_tutors[:10]}")
        logger.debug(f"unique_teaching_times: {unique_teaching_times[:10]}")
        
        self.subject_vocab = np.array(subject_vocab, dtype=np.bytes_)
        self.grade_vocab = np.array(grade_vocab, dtype=np.bytes_)
        
        self._tutor_lookup = tf.keras.layers.StringLookup(
            vocabulary=np.array(unique_tutors, dtype=np.bytes_), 
            mask_token=None, output_mode='int', name='tutor_lookup')
        self._tutor_embedding = tf.keras.layers.Embedding(input_dim=len(unique_tutors) + 2, output_dim=64, name='tutor_embedding')
        
        self._teaching_time_lookup = tf.keras.layers.StringLookup(
            vocabulary=np.array(unique_teaching_times, dtype=np.bytes_), 
            mask_token=None, output_mode='int', name='teaching_time_lookup')
        self._teaching_time_embedding = tf.keras.layers.Embedding(input_dim=len(unique_teaching_times) + 2, output_dim=64, name='teaching_time_embedding')
        
        self._subject_dense = tf.keras.layers.Dense(32, name='subject_dense')
        self._grade_dense = tf.keras.layers.Dense(32, name='grade_dense')
        
        self._experience_dense = tf.keras.layers.Dense(32, name='experience_dense')
        
        self._dense_layers = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(64*2 + 32 + 32 + 32,), name='tutor_dense_input'),
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.05), name='dense_1'),
            tf.keras.layers.Dropout(0.2, name='dropout_1'),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.05), name='dense_2'),
            tf.keras.layers.Dropout(0.2, name='dropout_2'),
            tf.keras.layers.Dense(32, name='dense_output')
        ], name='tutor_dense_layers')
        
        self.candidate_embeddings = None
    
    def call(self, inputs):
        experience = tf.cast(inputs.get('kinh_nghiem_giang_day', tf.zeros_like(inputs['ten_gia_su'], dtype=tf.float32)), tf.float32)
        experience_2d = tf.expand_dims(experience, axis=-1)
        
        tutor_indices = self._tutor_lookup(inputs['ten_gia_su'])
        teaching_time_indices = self._teaching_time_lookup(inputs['thoi_gian_day_hoc'])
        
        tutor_embedding = self._tutor_embedding(tutor_indices)
        teaching_time_embedding = self._teaching_time_embedding(teaching_time_indices)
        
        subject_inputs = [inputs.get(f'subject_pca_{i}', tf.zeros_like(inputs['ten_gia_su'], dtype=tf.float32)) for i in range(10)]
        subject_multi_hot = tf.stack(subject_inputs, axis=-1)
        subject_embedding = self._subject_dense(subject_multi_hot)
        
        grade_inputs = [inputs.get(f'grade_pca_{i}', tf.zeros_like(inputs['ten_gia_su'], dtype=tf.float32)) for i in range(5)]
        grade_multi_hot = tf.stack(grade_inputs, axis=-1)
        grade_embedding = self._grade_dense(grade_multi_hot)
        
        experience_embedding = self._experience_dense(experience_2d)
        
        concatenated = tf.concat([tutor_embedding, teaching_time_embedding, subject_embedding, grade_embedding, experience_embedding], axis=-1)
        output = self._dense_layers(concatenated)
        return output
    
    def get_config(self):
        config = {
            'unique_tutors': list(self._tutor_lookup.get_vocabulary()),
            'unique_teaching_times': list(self._teaching_time_lookup.get_vocabulary()),
            'subject_vocab': list(self.subject_vocab.astype(str)),
            'grade_vocab': list(self.grade_vocab.astype(str))
        }
        logger.debug(f"TutorModel get_config: {config}")
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(
            unique_tutors=config['unique_tutors'],
            unique_teaching_times=config['unique_teaching_times'],
            subject_vocab=config['subject_vocab'],
            grade_vocab=config['grade_vocab']
        )

class MaterialModel(tf.keras.Model):
    def __init__(self, unique_materials, subject_vocab, grade_vocab, material_type_vocab):
        super().__init__()
        
        unique_materials = [sanitize_name(x) for x in unique_materials if isinstance(x, str) and x and x.lower() != 'counter']
        subject_vocab = [sanitize_name(x) for x in subject_vocab if isinstance(x, str) and x and x.lower() != 'counter']
        grade_vocab = [sanitize_name(x) for x in grade_vocab if isinstance(x, str) and x and x.lower() != 'counter']
        material_type_vocab = [sanitize_name(x) for x in material_type_vocab if isinstance(x, str) and x and x.lower() != 'counter']
        
        if not subject_vocab:
            logger.warning("Empty subject_vocab, using ['unknown']")
            subject_vocab = ['unknown']
        if not grade_vocab:
            logger.warning("Empty grade_vocab, using ['unknown']")
            grade_vocab = ['unknown']
        if not unique_materials:
            logger.warning("Empty unique_materials, using ['unknown']")
            unique_materials = ['unknown']
        if not material_type_vocab:
            logger.warning("Empty material_type_vocab, using ['unknown']")
        logger.info(f"MaterialModel vocab sizes: materials={len(unique_materials)}, "
                    f"subjects={len(subject_vocab)}, grades={len(grade_vocab)}, types={len(material_type_vocab)}")
        logger.debug(f"subject_vocab: {subject_vocab[:10]}")
        logger.debug(f"grade_vocab: {grade_vocab[:10]}")
        logger.debug(f"material_type_vocab: {material_type_vocab[:10]}")
        logger.debug(f"unique_materials: {unique_materials[:10]}")
        
        self.subject_vocab = np.array(subject_vocab, dtype=np.bytes_)
        self.grade_vocab = np.array(grade_vocab, dtype=np.bytes_)
        self.material_type_vocab = np.array(material_type_vocab, dtype=np.bytes_)
        
        self._material_lookup = tf.keras.layers.StringLookup(
            vocabulary=np.array(unique_materials, dtype=np.bytes_), 
            mask_token=None, output_mode='int', name='material_lookup')
        self._material_embedding = tf.keras.layers.Embedding(input_dim=len(unique_materials) + 2, output_dim=128, name='material_embedding')
        
        self._subject_dense = tf.keras.layers.Dense(32, name='subject_dense')
        self._grade_dense = tf.keras.layers.Dense(32, name='grade_dense')
        self._type_dense = tf.keras.layers.Dense(32, name='type_dense')
        
        self._dense_layers = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128 + 32 + 32 + 32,), name='material_dense_input'),
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01), name='dense_1'),
            tf.keras.layers.Dropout(0.2, name='dropout_1'),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01), name='dense_2'),
            tf.keras.layers.Dropout(0.2, name='dropout_2'),
            tf.keras.layers.Dense(32, name='dense_output')
        ], name='material_dense_layers')
        
        self.candidate_embeddings = None
    
    def call(self, inputs):
        material_indices = self._material_lookup(inputs['ten_tai_lieu'])
        material_embedding = self._material_embedding(material_indices)
        
        subject_inputs = [inputs.get(f'subject_pca_{i}', tf.zeros_like(inputs['ten_tai_lieu'], dtype=tf.float32)) for i in range(10)]
        subject_multi_hot = tf.stack(subject_inputs, axis=-1)
        subject_embedding = self._subject_dense(subject_multi_hot)
        
        grade_inputs = [inputs.get(f'grade_pca_{i}', tf.zeros_like(inputs['ten_tai_lieu'], dtype=tf.float32)) for i in range(5)]
        grade_multi_hot = tf.stack(grade_inputs, axis=-1)
        grade_embedding = self._grade_dense(grade_multi_hot)
        
        type_inputs = [inputs.get(f'material_type_pca_{i}', tf.zeros_like(inputs['ten_tai_lieu'], dtype=tf.float32)) for i in range(5)]
        type_multi_hot = tf.stack(type_inputs, axis=-1)
        type_embedding = self._type_dense(type_multi_hot)
        
        concatenated = tf.concat([material_embedding, subject_embedding, grade_embedding, type_embedding], axis=-1)
        output = self._dense_layers(concatenated)
        return output
    
    def get_config(self):
        config = {
            'unique_materials': list(self._material_lookup.get_vocabulary()),
            'subject_vocab': list(self.subject_vocab.astype(str)),
            'grade_vocab': list(self.grade_vocab.astype(str)),
            'material_type_vocab': list(self.material_type_vocab.astype(str))
        }
        logger.debug(f"MaterialModel get_config: {config}")
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(
            unique_materials=config['unique_materials'],
            subject_vocab=config['subject_vocab'],
            grade_vocab=config['grade_vocab'],
            material_type_vocab=config['material_type_vocab']
        )

class RecommendationModel(tfrs.Model):
    def __init__(self, student_features, course_features, tutor_features, material_features,
                 student_course_train, student_tutor_train, student_material_train,
                 subject_vocab, grade_vocab, material_type_vocab, teaching_time_vocab, bruteforce_data_path=None):
        super().__init__()
        
        unique_schools = list(set(
            [sanitize_name(x) for x in student_features['truong_hoc_hien_tai'].unique() if str(x).lower() != 'counter'] +
            [sanitize_name(x) for x in student_course_train['truong_hoc_hien_tai'].unique() if str(x).lower() != 'counter'] +
            [sanitize_name(x) for x in student_tutor_train['truong_hoc_hien_tai'].unique() if str(x).lower() != 'counter'] +
            [sanitize_name(x) for x in student_material_train['truong_hoc_hien_tai'].unique() if str(x).lower() != 'counter']
        ))
        unique_goals = list(set(
            [sanitize_name(x) for x in student_features['muc_tieu_hoc'].unique() if str(x).lower() != 'counter'] +
            [sanitize_name(x) for x in student_course_train['muc_tieu_hoc'].unique() if str(x).lower() != 'counter'] +
            [sanitize_name(x) for x in student_tutor_train['muc_tieu_hoc'].unique() if str(x).lower() != 'counter'] +
            [sanitize_name(x) for x in student_material_train['muc_tieu_hoc'].unique() if str(x).lower() != 'counter']
        ))
        unique_learning_methods = list(set(
            [sanitize_name(x) for x in student_features['phuong_phap_hoc_yeu_thich'].unique() if str(x).lower() != 'counter'] +
            [sanitize_name(x) for x in student_course_train['phuong_phap_hoc_yeu_thich'].unique() if str(x).lower() != 'counter'] +
            [sanitize_name(x) for x in student_tutor_train['phuong_phap_hoc_yeu_thich'].unique() if str(x).lower() != 'counter'] +
            [sanitize_name(x) for x in student_material_train['phuong_phap_hoc_yeu_thich'].unique() if str(x).lower() != 'counter']
        ))
        
        unique_centers = [sanitize_name(x) for x in course_features['ten_trung_tam'].unique() if str(x).lower() != 'counter']
        unique_course_methods = [sanitize_name(x) for x in course_features['phuong_phap_hoc'].unique() if str(x).lower() != 'counter']
        unique_tutors = [sanitize_name(x) for x in tutor_features['ten_gia_su'].unique() if str(x).lower() != 'counter']
        unique_teaching_times = [sanitize_name(x) for x in teaching_time_vocab if x.lower() != 'counter']
        unique_materials = [sanitize_name(x) for x in material_features['ten_tai_lieu'].unique() if str(x).lower() != 'counter']
        
        self.student_model = StudentTower(unique_schools, unique_goals, unique_learning_methods, subject_vocab, grade_vocab)
        self.course_model = CourseModel(unique_centers, unique_course_methods, subject_vocab, grade_vocab)
        self.tutor_model = TutorModel(unique_tutors, unique_teaching_times, subject_vocab, grade_vocab)
        self.material_model = MaterialModel(unique_materials, subject_vocab, grade_vocab, material_type_vocab)
        
        self.student_course_train = create_tf_dataset(student_course_train).batch(32).cache().prefetch(tf.data.AUTOTUNE)
        self.student_tutor_train = create_tf_dataset(student_tutor_train).batch(32).cache().prefetch(tf.data.AUTOTUNE)
        self.material_train = create_tf_dataset(student_material_train).batch(32).cache().prefetch(tf.data.AUTOTUNE)
        
        self.course_features = course_features
        self.tutor_features = tutor_features
        self.material_features = material_features
        
        course_columns = [
            'id_trung_tam', 'ten_trung_tam', 'phuong_phap_hoc', 'thoi_gian', 'chi_phi', 'dia_chi', 'danh_gia'
        ] + [f'subject_pca_{i}' for i in range(10)] + [f'grade_pca_{i}' for i in range(5)]
        available_course_columns = [col for col in course_columns if col in course_features.columns]
        course_dataset = tf.data.Dataset.from_tensor_slices({
            k: v for k, v in dict(course_features).items() if k in available_course_columns
        }).batch(32)
        course_embeddings = []
        course_ids = []
        for batch in course_dataset:
            embeddings = self.course_model(batch)
            ids = tf.strings.join(['course_', tf.strings.as_string(batch['id_trung_tam'])])
            course_embeddings.append(embeddings)
            course_ids.append(ids)
        self.course_model.candidate_embeddings = {
            'embeddings': tf.concat(course_embeddings, axis=0),
            'identifiers': tf.concat(course_ids, axis=0)
        }
        logger.info(f"Course candidate embeddings shape: {self.course_model.candidate_embeddings['embeddings'].shape}")
        
        tutor_columns = [
            'id_gia_su', 'ten_gia_su', 'thoi_gian_day_hoc', 'kinh_nghiem_giang_day'
        ] + [f'subject_pca_{i}' for i in range(10)] + [f'grade_pca_{i}' for i in range(5)]
        available_tutor_columns = [col for col in tutor_columns if col in tutor_features.columns]
        tutor_dataset = tf.data.Dataset.from_tensor_slices({
            k: v for k, v in dict(tutor_features).items() if k in available_tutor_columns
        }).batch(32)
        tutor_embeddings = []
        tutor_ids = []
        for batch in tutor_dataset:
            embeddings = self.tutor_model(batch)
            ids = tf.strings.join(['tutor_', tf.strings.as_string(batch['id_gia_su'])])
            tutor_embeddings.append(embeddings)
            tutor_ids.append(ids)
        self.tutor_model.candidate_embeddings = {
            'embeddings': tf.concat(tutor_embeddings, axis=0),
            'identifiers': tf.concat(tutor_ids, axis=0)
        }
        logger.info(f"Tutor candidate embeddings shape: {self.tutor_model.candidate_embeddings['embeddings'].shape}")
        
        material_columns = [
            'id_tai_lieu', 'ten_tai_lieu'
        ] + [f'subject_pca_{i}' for i in range(10)] + [f'grade_pca_{i}' for i in range(5)] + [f'material_type_pca_{i}' for i in range(5)]
        available_material_columns = [col for col in material_columns if col in material_features.columns]
        material_dataset = tf.data.Dataset.from_tensor_slices({
            k: v for k, v in dict(material_features).items() if k in available_material_columns
        }).batch(32)
        material_embeddings = []
        material_ids = []
        for batch in material_dataset:
            embeddings = self.material_model(batch)
            ids = tf.strings.join(['material_', tf.strings.as_string(batch['id_tai_lieu'])])
            material_embeddings.append(embeddings)
            material_ids.append(ids)
        self.material_model.candidate_embeddings = {
            'embeddings': tf.concat(material_embeddings, axis=0),
            'identifiers': tf.concat(material_ids, axis=0)
        }
        logger.info(f"Material candidate embeddings shape: {self.material_model.candidate_embeddings['embeddings'].shape}")
        
        # Initialize retrieval tasks
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
        
        # Build BruteForce layer
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
            self.bruteforce = tfrs.layers.factorized_top_k.BruteForce(k=10, name='bruteforce')
            self.bruteforce.index(all_embeddings, identifiers=all_identifiers)
            sample_query = tf.random.normal([1, 32])
            self.bruteforce(sample_query)
            logger.info("BruteForce layer built successfully")
        
        # Build metrics with realistic data
        self.build_metrics()
        
        self.subject_vocab = subject_vocab
        self.grade_vocab = grade_vocab
        self.material_type_vocab = material_type_vocab
        self.teaching_time_vocab = teaching_time_vocab
        self.bruteforce_data_path = bruteforce_data_path
        self._bruteforce_saved = False  # Flag to track BruteForce save status
    
    def build_metrics(self):
        logger.info("Building FactorizedTopK, Precision, and Recall metrics...")
        sample_batch = None
        for batch in self.student_course_train.take(1):
            sample_batch = batch
            break
        if sample_batch is None:
            logger.warning("No training data available for metrics building, using random data")
            sample_query = tf.random.normal([1, 32])
        else:
            sample_query = self.student_model(sample_batch)

        for task, task_name, candidate_embeddings in [
            (self.course_task, 'course_task', self.course_model.candidate_embeddings['embeddings']),
            (self.tutor_task, 'tutor_task', self.tutor_model.candidate_embeddings['embeddings']),
            (self.material_task, 'material_task', self.material_model.candidate_embeddings['embeddings'])
        ]:
            try:
                logger.info(f"{task_name} candidate embeddings shape: {candidate_embeddings.shape}")
                if candidate_embeddings.shape[0] == 0:
                    logger.warning(f"No candidate embeddings for {task_name}, skipping metrics build")
                    continue
                candidate_batch = candidate_embeddings[:min(32, candidate_embeddings.shape[0])]
                scores = tf.matmul(sample_query, candidate_batch, transpose_b=True)
                task.metrics[0].update_state(scores)  # Update FactorizedTopK
                logger.info(f"{task_name} FactorizedTopK metrics built successfully")
            except Exception as e:
                logger.error(f"Failed to build {task_name} FactorizedTopK metrics: {str(e)}")

        # Add Precision and Recall metrics to tasks
        self.course_task.metrics.append(PrecisionAtK(k=10, name='precision_at_10'))
        self.course_task.metrics.append(RecallAtK(k=10, name='recall_at_10'))
        self.tutor_task.metrics.append(PrecisionAtK(k=10, name='precision_at_10'))
        self.tutor_task.metrics.append(RecallAtK(k=10, name='recall_at_10'))
        self.material_task.metrics.append(PrecisionAtK(k=10, name='recall_at_10'))
        self.material_task.metrics.append(RecallAtK(k=10, name='recall_at_10'))

    def evaluate_test(self, test_dataset, task_name, id_key, candidate_identifiers):
        logger.info(f"Evaluating {task_name} on test dataset...")
        precision_metric = PrecisionAtK(k=10, name=f'{task_name}_precision_at_10')
        recall_metric = RecallAtK(k=10, name=f'{task_name}_recall_at_10')

        for batch in test_dataset:
            try:
                student_embeddings = self.student_model(batch)
                _, top_k_ids = self.bruteforce(student_embeddings)
                ground_truth_ids = tf.strings.join([task_name + '_', tf.strings.as_string(batch[id_key])])
                precision_metric.update_state(ground_truth_ids, top_k_ids)
                recall_metric.update_state(ground_truth_ids, top_k_ids)
            except Exception as e:
                logger.error(f"Error evaluating {task_name} batch: {str(e)}")
                continue

        precision_result = precision_metric.result().numpy()
        recall_result = recall_metric.result().numpy()
        logger.info(f"{task_name} - Precision@10: {precision_result:.4f}, Recall@10: {recall_result:.4f}")
        return precision_result, recall_result
        
    def call(self, inputs, training=False):
        student_embeddings = self.student_model(inputs)
        
        outputs = {}
        if 'id_trung_tam' in inputs:
            course_batch = {k: v for k, v in inputs.items() if k in self.course_features.columns}
            course_embeddings = self.course_model(course_batch)
            outputs['course_output'] = course_embeddings
        
        if 'id_gia_su' in inputs:
            tutor_batch = {k: v for k, v in inputs.items() if k in self.tutor_features.columns}
            tutor_embeddings = self.tutor_model(tutor_batch)
            outputs['tutor_output'] = tutor_embeddings
        
        if 'id_tai_lieu' in inputs:
            material_batch = {k: v for k, v in inputs.items() if k in self.material_features.columns}
            material_embeddings = self.material_model(material_batch)
            outputs['material_output'] = material_embeddings
        
        return outputs
    
    @tf.function
    def compute_loss(self, inputs, training=False):
        total_loss = 0.0
        batch_count = 0
        
        outputs = self.call(inputs, training=training)
        student_embeddings = self.student_model(inputs)
        
        if 'course_output' in outputs:
            course_embeddings = outputs['course_output']
            loss = self.course_task(
                query_embeddings=student_embeddings,
                candidate_embeddings=course_embeddings,
                compute_metrics=training
            )
            total_loss += loss
            batch_count += 1
            if training:
                logger.debug("Computed course_task metrics")
        
        if 'tutor_output' in outputs:
            tutor_embeddings = outputs['tutor_output']
            loss = self.tutor_task(
                query_embeddings=student_embeddings,
                candidate_embeddings=tutor_embeddings,
                compute_metrics=training
            )
            total_loss += loss
            batch_count += 1
            if training:
                logger.debug("Computed tutor_task metrics")
        
        if 'material_output' in outputs:
            material_embeddings = outputs['material_output']
            loss = self.material_task(
                query_embeddings=student_embeddings,
                candidate_embeddings=material_embeddings,
                compute_metrics=training
            )
            total_loss += loss
            batch_count += 1
            if training:
                logger.debug("Computed material_task metrics")
        
        return total_loss / tf.cast(batch_count, tf.float32) if batch_count > 0 else total_loss
    
    def save_bruteforce_data(self, path):
        if self.bruteforce is None:
            logger.warning("BruteForce layer not initialized, skipping save.")
            return
        if self._bruteforce_saved:
            logger.warning(f"BruteForce data already saved to {path}, skipping redundant save.")
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
            self._bruteforce_saved = True
            logger.info(f"BruteForce data saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save BruteForce data: {str(e)}")
    
    def load_bruteforce_data(self, path):
        try:
            data = np.load(path)
            embeddings = tf.convert_to_tensor(data['embeddings'], dtype=tf.float32)
            identifiers = tf.convert_to_tensor(data['identifiers'], dtype=tf.string)
            self.bruteforce = tfrs.layers.factorized_top_k.BruteForce(k=10, name='bruteforce')
            self.bruteforce.index(embeddings, identifiers=identifiers)
            sample_query = tf.random.normal([1, 32])
            self.bruteforce(sample_query)
            self._bruteforce_saved = False  # Reset flag after loading
            logger.info(f"BruteForce data loaded and built from {path}")
        except Exception as e:
            logger.error(f"Failed to load BruteForce data from {path}: {str(e)}")
    
    def get_config(self):
        config = {
            'subject_vocab': self.subject_vocab,
            'grade_vocab': self.grade_vocab,
            'material_type_vocab': self.material_type_vocab,
            'teaching_time_vocab': self.teaching_time_vocab,
            'bruteforce_data_path': self.bruteforce_data_path
        }
        logger.debug(f"RecommendationModel get_config: {config}")
        return config
    
    @classmethod
    def from_config(cls, config):
        student_features = pd.DataFrame()
        course_features = pd.DataFrame()
        tutor_features = pd.DataFrame()
        material_features = pd.DataFrame()
        student_course_train = pd.DataFrame()
        student_tutor_train = pd.DataFrame()
        student_material_train = pd.DataFrame()
        
        return cls(
            student_features=student_features,
            course_features=course_features,
            tutor_features=tutor_features,
            material_features=material_features,
            student_course_train=student_course_train,
            student_tutor_train=student_tutor_train,
            student_material_train=student_material_train,
            subject_vocab=config['subject_vocab'],
            grade_vocab=config['grade_vocab'],
            material_type_vocab=config['material_type_vocab'],
            teaching_time_vocab=config['teaching_time_vocab'],
            bruteforce_data_path=config.get('bruteforce_data_path')
        )
    
    def save(self, filepath, overwrite=True, include_optimizer=True, save_format='tf', signatures=None, options=None):
        """Custom save method to handle complex model serialization."""
        logger.info(f"Attempting to save full model to {filepath}")
        self._bruteforce_saved = False  # Reset flag before saving
        for test_dataset in [self.student_course_train, self.student_tutor_train, self.material_train]:
            for batch in test_dataset.take(1):
                    self.build_metrics()
                    break
            try:
                super().save(filepath, overwrite=overwrite, include_optimizer=include_optimizer, 
                            save_format=save_format, signatures=signatures, options=options)
                logger.info(f"Full model saved successfully to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save full model: {str(e)}")
        try:
            super().save(filepath, overwrite=overwrite, include_optimizer=include_optimizer, 
                        save_format=save_format, signatures=signatures, options=options)
            logger.info(f"Full model saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save full model: {str(e)}")
            logger.info("Attempting to save sub-models individually...")
            
            os.makedirs(filepath, exist_ok=True)
            
            sub_model_paths = {
                'student_model': os.path.join(filepath, 'student_model'),
                'course_model': os.path.join(filepath, 'course_model'),
                'tutor_model': os.path.join(filepath, 'tutor_model'),
                'material_model': os.path.join(filepath, 'material_model')
            }
            
            for model_name, model_path in sub_model_paths.items():
                try:
                    model = getattr(self, model_name)
                    model.save(model_path, overwrite=overwrite, include_optimizer=include_optimizer, 
                              save_format=save_format, options=options)
                    logger.info(f"Saved {model_name} to {model_path}")
                except Exception as sub_e:
                    logger.error(f"Failed to save {model_name}: {str(sub_e)}")
            
            bruteforce_path = os.path.join(filepath, 'bruteforce_data.npz')
            try:
                self.save_bruteforce_data(bruteforce_path)
            except Exception as bf_e:
                logger.error(f"Failed to save BruteForce data: {str(bf_e)}")
            
            config_path = os.path.join(filepath, 'model_config.json')
            try:
                config = self.get_config()
                with open(config_path, 'w') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved model configuration to {config_path}")
            except Exception as config_e:
                logger.error(f"Failed to save model configuration: {str(config_e)}")
            
            logger.info(f"Sub-model saving completed. Check {filepath} for saved components.")
