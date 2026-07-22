import cv2
import face_recognition
import pickle

with open("face_model.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    for (top, right, bottom, left), encoding in zip(locations, encodings):

        matches = face_recognition.compare_faces(
            known_encodings,
            encoding,
            tolerance=0.5
        )

        name = "Unknown"

        if True in matches:
            first_match = matches.index(True)
            name = known_names[first_match]

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

        cv2.putText(frame,
                    name,
                    (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()