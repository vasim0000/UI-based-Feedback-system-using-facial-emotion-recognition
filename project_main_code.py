import cv2
import os
import numpy as np

video_file = 'zoom_video.mp4'
edited_list=[]

cap = cv2.VideoCapture(video_file)
ret,frame = cap.read()
x_coordinate,y_coordinate,channel=frame.shape

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

counter=0

os.makedirs('frames')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        black_img=np.zeros_like(frame[y:y+h, x:x+w])
        edited_img =cv2.bitwise_xor(frame[y:y+h, x:x+w], black_img)
        edited_list.append(edited_img)

    cv2.imshow('Video',frame)

    if counter%24==0:
        cv2.imwrite('path\\test_images%d.jpg' % counter, frame)

    counter+=1

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()


counter=0
os.makedirs('faces')
for x in edited_list:
    cv2.imwrite('faces//face%d.jpg' %counter,x)
    counter+=1
cv2.destroyAllWindows()
