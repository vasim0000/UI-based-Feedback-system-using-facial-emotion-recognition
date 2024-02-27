import cv2
import os
import face_recognition
from collections import defaultdict

output_directory = 'path_to_output_folder'
os.makedirs(output_directory, exist_ok=True)

video_path = 'zoom_HD.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

frames_to_skip = int(fps) - 1

known_persons = defaultdict(list)

def compare_face_encodings(encodings, face_encoding):
    return face_recognition.compare_faces(encodings, face_encoding, tolerance=0.6)  # Adjust the tolerance as needed

frame_number = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break 

    frame_number += 1
    
    if frame_number % (frames_to_skip + 1) != 0:
        continue

    face_locations = face_recognition.face_locations(frame)

    for face_location in face_locations:
        top, right, bottom, left = face_location

        top = max(0, top - int((bottom - top) * 0.25))

        face_encoding = face_recognition.face_encodings(frame, [face_location])[0]

        found_match = False
        for person, encodings in known_persons.items():
            if any(compare_face_encodings(encodings, face_encoding)):
                found_match = True
                person_folder = os.path.join(output_directory, f'person_{person}')
                break

        if not found_match:
            person = len(known_persons) + 1
            known_persons[person].append(face_encoding)
            person_folder = os.path.join(output_directory, f'person_{person}')
            os.makedirs(person_folder, exist_ok=True)

        head_roi = frame[top:bottom, left:right]

        cv2.imwrite(os.path.join(person_folder, f'frame_{frame_number}_head_{person}.jpg'), head_roi)

cap.release()
cv2.destroyAllWindows()
