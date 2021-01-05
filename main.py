import os
from datetime import datetime

import cv2
import face_recognition
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 600)
cap.set(4, 600)
cap.set(10, 150)

path = 'ImagesPp'
images = []
names = []
myList = os.listdir(path)

for img_name in myList:
    img = cv2.imread(f'{path}/{img_name}')
    images.append(img)
    names.append(os.path.splitext(img_name)[0])


# print(names)


def encoding(images_lis):
    encoded_list = []
    for im in images_lis:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(im)[0]
        encoded_list.append(encode)
    return encoded_list


def present(person_name):
    with open("Attendance.csv", 'r+') as f:
        data_list = f.readlines()
        name_list = []
        for line in data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if person_name not in name_list:
            now = datetime.now()
            dt_str = now.strftime('%H: %M')
            f.writelines(f'\n {person_name}, {dt_str}')


encodeKnownFaces = encoding(images)

while True:
    _, frame = cap.read()
    imgs = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgs)
    encodeCurFrame = face_recognition.face_encodings(imgs, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeKnownFaces, encodeFace)
        face_dis = face_recognition.face_distance(encodeKnownFaces, encodeFace)
        match_index = np.argmin(face_dis)

        if face_dis[match_index] < 0.50:
            name = names[match_index].upper()
            present(name)
        else:
            name = 'Unknown'
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED)
        cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

    cv2.imshow('Cam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
