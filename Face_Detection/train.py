import face_recognition
import os
import pickle

KNOWN_FACE_DIR = "known_faces"

known_encodings = []
known_names = []

for person_name in os.listdir(KNOWN_FACE_DIR):

    person_folder = os.path.join(KNOWN_FACE_DIR, person_name)

    if not os.path.isdir(person_folder):
        continue

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        image = face_recognition.load_image_file(image_path)

        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person_name)

            print(f"Loaded : {person_name} -> {image_name}")

with open("face_model.pkl", "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print("\nTraining Completed Successfully.")