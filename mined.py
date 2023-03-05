import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'D:\\Nirma\\MineD\\dataset1\\img'
images = []
Names = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    Names.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('D:\\Nirma\\MineD\\Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
 

encodeListKnown = findEncodings(images)
 
cap = cv2.VideoCapture('D:\\Nirma\\MineD\\dataset1\\Video_clip\\VID_20230302_170157.mp4')
cnt=0
while True:
    cap.set(cv2.CAP_PROP_POS_MSEC,(cnt*1000))
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    cnt+=1 
    facesCurFrame = face_recognition.face_locations(imgS)
    #top,left,right,bottom 
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
     
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex] and faceDis[matchIndex]<0.45:
            name = Names[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(255,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 255), cv2.FILLED)
            cv2.putText(img, "Unknown" , (x1+6, y2-6),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Webcam',img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    cv2.waitKey(1)