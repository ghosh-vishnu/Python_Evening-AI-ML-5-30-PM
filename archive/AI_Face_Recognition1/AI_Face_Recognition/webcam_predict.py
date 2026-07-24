"""
webcam_predict.py
-------------------
Runs real-time face recognition using your webcam and the trained model.

Usage:
    python webcam_predict.py
Press 'q' to quit.
"""

import cv2
import os
import pickle

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


def run_webcam_prediction():
    recognizer, id_to_name = load_model()
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Could not open webcam. Check camera permissions/index.")

    print("[INFO] Starting webcam recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            label_id, confidence = recognizer.predict(face_roi)

            if confidence <= CONFIDENCE_THRESHOLD:
                name = id_to_name.get(label_id, "Unknown")
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)

            text = f"{name} ({confidence:.0f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

        cv2.imshow("Webcam Face Recognition - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_prediction()
