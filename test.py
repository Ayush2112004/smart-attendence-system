import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import csv

# Load known faces and their names
path = r'C:\Users\ayush\OneDrive\Pictures'
images = []
classNames = []

for filename in os.listdir(path):
    if filename.endswith(('.jpg', '.png')):
        img = cv2.imread(f"{path}/{filename}")
        images.append(img)
        classNames.append(os.path.splitext(filename)[0])

def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encode_list.append(encodes[0])
    return encode_list

encodeListKnown = find_encodings(images)
print('Encoding Complete')

def mark_attendance(name):
    with open('attendance.csv', 'r+') as f:
        existing_data = f.readlines()
        names_logged = [line.split(',')[0] for line in existing_data]
        if name not in names_logged:
            now = datetime.now()
            dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'"{name}","{dt_string}"\n')


# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            mark_attendance(name)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
