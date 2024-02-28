import cv2
import os
import face_recognition
from collections import defaultdict
from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np

# Output directory for saving heads
output_directory = 'path_to_output_folder'
os.makedirs(output_directory, exist_ok=True)

# Open the video file
video_path = 'zoom_HD.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error opening video file.")
    exit()

# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the number of frames to skip to achieve one frame per second
frames_to_skip = int(fps) - 1

# Dictionary to store face encodings for each person
known_persons = defaultdict(list)
person_emotions = defaultdict(list)

# Function to compare face encodings
def compare_face_encodings(encodings, face_encoding):
    return face_recognition.compare_faces(encodings, face_encoding, tolerance=0.6)  # Adjust the tolerance as needed

# Function to get dominant emotion
def get_dominant_emotion(face_image_path):
    result = DeepFace.analyze(face_image_path, actions=['emotion'], enforce_detection=False)
    return result[0]['dominant_emotion']

# Iterate through each frame in the video
frame_number = 0
while True:
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if we have reached the end of the video

    frame_number += 1

    # Skip frames based on the calculated frames_to_skip
    if frame_number % (frames_to_skip + 1) != 0:
        continue

    # Find face locations in the frame
    face_locations = face_recognition.face_locations(frame)

    # Iterate through detected faces
    for face_location in face_locations:
        top, right, bottom, left = face_location

        # Extend the region to include more of the head
        top = max(0, top - int((bottom - top) * 0.25))

        # Extract face encoding
        face_encoding = face_recognition.face_encodings(frame, [face_location])[0]

        # Check if this face matches any known person
        found_match = False
        for person, encodings in known_persons.items():
            if any(compare_face_encodings(encodings, face_encoding)):
                found_match = True
                person_folder = os.path.join(output_directory, f'person_{person}')
                break

        # If not found, create a new person
        if not found_match:
            person = len(known_persons) + 1
            known_persons[person].append(face_encoding)
            person_folder = os.path.join(output_directory, f'person_{person}')
            os.makedirs(person_folder, exist_ok=True)

        # Crop the head from the frame
        head_roi = frame[top:bottom, left:right]

        # Save the head as an image file in the person-specific folder
        image_path = os.path.join(person_folder, f'frame_{frame_number}_head_{person}.jpg')
        cv2.imwrite(image_path, head_roi)

        # Get the dominant emotion for the head
        dominant_emotion = get_dominant_emotion(image_path)

        # Save the emotion for the person
        person_emotions[person].append(dominant_emotion)

# Release the video capture object
cap.release()
cv2.destroyAllWindows()

# Visualize each person's dominant emotion with a pie chart, histogram, and face image
person_count=1
for person, emotions in person_emotions.items():
    plt.figure(figsize=(15, 5))

    # Pie Chart
    plt.subplot(1, 3, 1)
    labels, counts = zip(*[(emotion, emotions.count(emotion)) for emotion in set(emotions)])
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(f'Person {person} - Dominant Emotion Pie Chart')

    # Histogram
    plt.subplot(1, 3, 2)
    sorted_emotions = sorted(set(emotions))
    label_mapping = {emotion: i for i, emotion in enumerate(sorted_emotions)}
    numeric_labels = [label_mapping[emotion] for emotion in emotions]
    hist, bins = np.histogram(numeric_labels, bins=len(sorted_emotions))
    plt.bar(range(len(sorted_emotions)), hist, tick_label=sorted_emotions, edgecolor='black')
    plt.title(f'Person {person} - Dominant Emotion Histogram')


    # Display the person's face image
    plt.subplot(1, 3, 3)
    frame_number=30
    size = len(person_folder)  # text length
    replacement = str(person_count)  # replace with this
    person_count+=1
    person_folder = person_folder.replace(person_folder[size - 1:], replacement)


    face_image_path = os.path.join(person_folder, f'frame_{frame_number}_head_{person}.jpg')

    print(face_image_path)

    if os.path.exists(face_image_path):
        print(f"Loading face image from: {face_image_path}")
        face_image = cv2.imread(face_image_path)
        if face_image is not None:
            print("Face image loaded successfully.")
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            plt.imshow(face_image)
            plt.title(f'Person {person} - Face Image')
        else:
            print("Failed to load face image.")
            plt.title(f'Person {person} - Face Image (Failed to Load)')
    else:
        print("Face image not found.")
        plt.title(f'Person {person} - Face Image (Not Found)')

    plt.tight_layout()
    plt.show()

os.rmdir('path_to_output_folder')
