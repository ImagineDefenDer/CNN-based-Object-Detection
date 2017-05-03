# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 21:54:25 2016

@author: lenovo
"""

import cv2
import cPickle
from matplotlib import pyplot as plt

def draw():
    f = open("final_result.pkl","rb")
    result = cPickle.load(f)
    f.close()
    
    oimg = cv2.imread('detec_2.jpg')
    
    #font=cv.InitFont(cv.CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0, 3, 8)
    
    for i in range(len(result)):

        LU = (result[i][0][0],result[i][0][1])
        RD = (result[i][0][0]+result[i][0][2],result[i][0][1]+result[i][0][3])
        cv2.rectangle(oimg, LU, RD, (255,0,0), 2)
        cv2.putText(oimg, '%s %f'%(result[i][1],result[i][2]), LU,cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    plt.imshow(oimg)
    
draw()