"""
train_model.py
----------------
Trains an LBPH (Local Binary Patterns Histograms) face recognizer using
the images in dataset/<person_name>/*.jpg and saves:
    model/trained_model.yml   -> the trained recognizer
    model/labels.pickle       -> mapping of numeric label -> person name

Usage:
    python train_model.py
"""

import cv2
import os
import pickle
import numpy as np

DATASET_DIR = "dataset"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pickle")


def load_dataset():
    faces = []
    labels = []
    label_map = {}   # name -> numeric id
    current_id = 0

    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(f"Dataset folder '{DATASET_DIR}' not found.")

    for person_name in sorted(os.listdir(DATASET_DIR)):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        if person_name not in label_map:
            label_map[person_name] = current_id
            current_id += 1
        label_id = label_map[person_name]

        for file_name in os.listdir(person_dir):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            file_path = os.path.join(person_dir, file_name)

            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] Could not read {file_path}, skipping.")
                continue

            faces.append(img)
            labels.append(label_id)

    return faces, labels, label_map


def train_model():
    print("[INFO] Loading dataset...")
    faces, labels, label_map = load_dataset()

    if len(faces) == 0:
        raise ValueError(
            "No training images found. Run capture_images.py first."
        )

    print(f"[INFO] Loaded {len(faces)} images across {len(label_map)} person(s).")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("[INFO] Training recognizer...")
    recognizer.train(faces, np.array(labels))

    os.makedirs(MODEL_DIR, exist_ok=True)
    recognizer.save(MODEL_PATH)

    # Save label_map reversed so predict scripts can go id -> name
    id_to_name = {v: k for k, v in label_map.items()}
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(id_to_name, f)

    print(f"[INFO] Model saved to '{MODEL_PATH}'")
    print(f"[INFO] Labels saved to '{LABELS_PATH}'")
    print(f"[INFO] Known people: {list(label_map.keys())}")


if __name__ == "__main__":
    train_model()
