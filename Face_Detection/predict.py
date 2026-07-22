import face_recognition
import pickle

# Load Model
with open("face_model.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

# Test Image
image = face_recognition.load_image_file("test/test.jpg")

locations = face_recognition.face_locations(image)
encodings = face_recognition.face_encodings(image, locations)

for encoding in encodings:

    matches = face_recognition.compare_faces(
        known_encodings,
        encoding,
        tolerance=0.5
    )

    name = "Unknown"

    if True in matches:
        first_match = matches.index(True)
        name = known_names[first_match]

    print("Prediction :", name)