import h5py
with h5py.File('./recommendation_model_weights.weights.h5', 'r') as f:
    print(list(f.keys()))
    print(list(f['tutor_model'].keys()) if 'tutor_model' in f else "No tutor_model")
