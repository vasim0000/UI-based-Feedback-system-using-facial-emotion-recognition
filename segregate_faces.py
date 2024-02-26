import os
import face_recognition
from collections import defaultdict
import shutil

def organize_images(input_folder, output_folder):
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Load face encodings for all images in the input folder
    known_face_encodings = []
    known_person_ids = []

    for file_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, file_name)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            # Assuming the first face encoding represents the person
            known_face_encodings.append(face_encodings[0])
            # Extract the person ID from the image name
            known_person_ids.append(int(''.join(filter(str.isdigit, file_name))))

    # Group images by person
    images_by_person = defaultdict(list)

    for file_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, file_name)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        if face_encodings:
            # Compare the face encoding with known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
            if any(matches):
                person_id = known_person_ids[matches.index(True)]
                images_by_person[person_id].append(image_path)

    # Create output folders for each person
    for person_id, images in images_by_person.items():
        person_output_folder = os.path.join(output_folder, f"Person_{person_id}")
        os.makedirs(person_output_folder, exist_ok=True)

        # Organize images into respective folders
        for image_path in images:
            shutil.copy(image_path, person_output_folder)

if __name__ == "__main__":
    # Replace 'input_folder' and 'output_folder' with your actual paths
    input_folder = "faces"
    output_folder = "out"

    # Create folders and organize images using face recognition
    organize_images(input_folder, output_folder)
