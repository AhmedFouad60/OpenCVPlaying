#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 06:40:42 2018

@author: foush
"""
import cv2
detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0); #open the camera 
Id=raw_input('enter the id')
sampleNum=0
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #incrementing sample number
        sampleNum += 1
        #saving the captured face in the dataset folder
        cv2.imwrite("dataSet/user."+Id+'.'+str(sampleNum) + ".jpg",gray[y:y+h,x:x+w])
        cv2.imshow('frame',img)
        #wait for 100 miliseconds
    if cv2.waitKey(100)& 0xFF== ord('q'):
        break
    elif sampleNum>20:
        break
cam.release()
cv2.destroyAllWindows()
    

