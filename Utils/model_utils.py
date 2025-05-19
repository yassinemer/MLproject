import os
import pickle
from Utils.config import MODEL_DIR

def load_pickle(file_name):
    file_path = os.path.join(MODEL_DIR, file_name)
    with open(file_path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, file_name):
    file_path = os.path.join(MODEL_DIR, file_name)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)
