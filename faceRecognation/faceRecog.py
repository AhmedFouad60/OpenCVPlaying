#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 06:25:47 2018

create dataset -> dataset creator
train the recognizer -> trainner
detector -> detector

@author: foush
"""
import cv2
import numpy as np

recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
detector=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")



cam = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret,im=cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray,1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(255,0,0),2)
        Id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        
        if( conf < 50):
            if(Id==1):
                Id="Foush"
            elif(Id==2):
                Id="Artogol"
            elif(Id==20):
                Id="Tag"
            elif(Id==4):
                Id="ahmed fouad"
            elif(Id==10):
                Id="ahmed fouad"
        else:
            Id="Unknown"
        fontface = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 1
        fontcolor = (255, 255, 0)
        cv2.putText(im, str(Id), (x,y+h), fontface, fontscale, fontcolor) 
    cv2.imshow('im',im)
    if cv2.waitKey(100)& 0xFF== ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


