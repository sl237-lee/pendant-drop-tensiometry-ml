import pickle
from pathlib import Path

def save_dataset(dataset, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved {len(dataset)} shapes")

def load_dataset(filepath):
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    return dataset
