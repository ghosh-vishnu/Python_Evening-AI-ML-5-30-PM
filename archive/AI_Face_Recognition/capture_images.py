"""
capture_images.py
------------------
Captures face images from your webcam and saves them into
dataset/<person_name>/ so they can later be used to train the model.

Usage:
    python capture_images.py
"""

import cv2
import os

# Number of face images to capture per person
NUM_SAMPLES = 60

# Where the captured faces are stored
DATASET_DIR = "dataset"

# Haar Cascade for face detection (ships with OpenCV)
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def capture_images(person_name: str, num_samples: int = NUM_SAMPLES):
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        raise IOError("Failed to load Haar Cascade classifier.")

    save_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Could not open webcam. Check camera permissions/index.")

    print(f"[INFO] Capturing images for '{person_name}'. Look at the camera...")
    print("[INFO] Press 'q' to quit early.")

    count = 0
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            file_path = os.path.join(save_dir, f"{person_name}_{count}.jpg")
            cv2.imwrite(file_path, face_img)

            # Draw rectangle + progress for visual feedback
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, f"Samples: {count}/{num_samples}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            break  # only take one face per frame

        cv2.imshow("Capturing Faces - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Saved {count} images to '{save_dir}'.")


if __name__ == "__main__":
    name = input("Enter the person's name (used as folder/label): ").strip()
    if not name:
        print("[ERROR] Name cannot be empty.")
    else:
        capture_images(name)
