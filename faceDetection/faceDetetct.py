import cv2
import numpy as np
"""
Spyder Editor

This is a temporary script file.
"""
faceDetect=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cam=cv2.VideoCapture(0)
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.imshow("Face",img);
    if cv2.waitKey(1) & 0xFF == ord('q'):
           break
cam.release()
cv2.destoryAllWindows()

    


