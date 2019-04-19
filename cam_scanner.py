#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 07:45:05 2019

@author: qwe
"""

from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import math
#ap = argparse.ArgumentParser()
import sys
import cv2  
  
# np is an alias pointing to numpy library 
import numpy as np 
  
  
# capture frames from a camera 
cap = cv2.VideoCapture(0) 
  
def mapp(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew 
# loop runs if capturing has been initialized 
while(1): 
  
    # reads frames from a camera 
    ret, frame = cap.read() 
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow("image",frame)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    cv2.imshow("blur",blur)
    
    edge=cv2.Canny(gray,75,150)
    
   
    cv2.imshow("edged-image",edge)
    
   
    img,contours,hierarchy=cv2.findContours(edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=cv2.contourArea,reverse=True)
    for x in contours:
        arclen=cv2.arcLength(x,True)
        approx=cv2.approxPolyDP(x,0.25*arclen,True)
        
        if len(approx)==4:
            d=approx
            break
    
    pt1=mapp(d)
    pt2=np.float32([[0,0],[600,0],[600,600],[0,600]])
    scanned=cv2.getPerspectiveTransform(pt1,pt2)
    scanned=cv2.warpPerspective(frame,scanned,(600,600))
    
    warped = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset = 10, method = "gaussian")
    scanned = (warped > T).astype("uint8") * 255
    cv2.imshow("scanned image",scanned)
    k = cv2.waitKey(5) & 0xFF
    if k == 27: 
        break
    
    
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  