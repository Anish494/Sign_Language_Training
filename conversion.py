import numpy as np
import json

def npy_to_json(npy_path, json_path):
    data = np.load(npy_path, allow_pickle=True)
    with open(json_path, "w") as f:
        json.dump(data.tolist(), f, indent=2)

# Example usage
npy_to_json("X_test.npy", "X_test.json")