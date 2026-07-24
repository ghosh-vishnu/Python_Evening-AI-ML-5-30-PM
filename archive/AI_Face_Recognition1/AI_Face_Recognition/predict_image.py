"""
predict_image.py
------------------
Runs face recognition on a single image file using the trained model.

Usage:
    python predict_image.py path/to/image.jpg
"""

import cv2
import os
import pickle
import sys

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pickle")

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Lower distance = more confident match. Tune this if you get
# too many false positives/negatives.
CONFIDENCE_THRESHOLD = 70


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run train_model.py first."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    with open(LABELS_PATH, "rb") as f:
        id_to_name = pickle.load(f)

    return recognizer, id_to_name


def predict_image(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    recognizer, id_to_name = load_model()
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        print("[INFO] No faces detected in the image.")
        return

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        label_id, confidence = recognizer.predict(face_roi)

        # Lower confidence value = better match for LBPH
        if confidence <= CONFIDENCE_THRESHOLD:
            name = id_to_name.get(label_id, "Unknown")
        else:
            name = "Unknown"

        text = f"{name} ({confidence:.1f})"
        print(f"[RESULT] Detected: {text}")

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img, text, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

    output_path = "prediction_output.jpg"
    cv2.imwrite(output_path, img)
    print(f"[INFO] Annotated image saved to '{output_path}'")

    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_image.py path/to/image.jpg")
        sys.exit(1)

    predict_image(sys.argv[1])
