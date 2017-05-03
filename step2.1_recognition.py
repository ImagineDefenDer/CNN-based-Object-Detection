# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:14:25 2017

@author: admin
"""

import pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import cv2

def load_box():
    f = open("experimentdata\\generatebox_result.pkl","rb")
    box,cent,n = pickle.load(f)
    f.close()
    return box,cent,n

def overlap_rate(boxA,boxB):
    '''
    s_o is area of overlap
    s_a is area of boxA + boxB
    overlap_rate = s_o / s_a
    '''
    x1,y1,w1,h1 = boxA
    x2,y2,w2,h2 = boxB
       
    if (x1 > x2 + w2) or (x2 > x1 + w1) or (y1 > h2 + y2) or (y2 > h1 + y1):
        return 0
           
    else:
        X = sorted([x1,x1+w1,x2,x2+w2])
        Y = sorted([y1,y1+h1,y2,y2+h2])        
        s_o = (X[2] - X[1]) * (Y[2] - Y[1])
        s_a = w1 * h1 + w2 * h2 - s_o
        rate = s_o / s_a
        return rate

def clus(box,min_over):
    l = len(box)
    cluster_list = []
       
    for i in range(l):
        over_i = [overlap_rate(box[i],box[j]) for j in range(l)]
        cluster_list.append(over_i)
    
    select_list = []
    for t in range(l):
        s = [a for a in range(len(cluster_list[t])) if (cluster_list[t][a] > min_over)]
        select_list.append(s)    
        
    def panduan(A,a):
        for l in A:
            if a in l:
                return True
            
        return False
        
    S = []
    for m in range(l):
        if panduan(S,m):
            continue
        else:
            S.append(select_list[m])
        
    return S     

        
def clear(box):
    box_c = []
    
    def baohan(boxA,boxB):
        x1,y1,w1,h1 = boxA
        x2,y2,w2,h2 = boxB
        if (x1<x2) and (y1<y2) and (x1+w1>x2+w2) and (y1+h1>y2+h2):
            return 1
        elif (x1>x2) and (y1>y2) and (x1+w1<x2+w2) and (y1+h1<y2+h2):
            return -1
        else:
            return 0
    
    boxA = (2,2,10,10)
    boxB = (3,3,3,3)
    print (baohan(boxB,boxA))
    
    for b in box:
        box_c.append([baohan(b,bb) for bb in box])
    
    ac = np.array(box_c)
    aac = []
    for i in range(len(box_c)):
        ic = 0
        for s in ac[i]:
            if s == 1:
                ic+=1
        aac.append(ic)
    ss = [arg for arg in range(len(aac)) if aac[arg]>1]
    box_cc  = []  
    for i in range(len(box)):
        if i in ss:
            continue
        else:
            box_cc.append(box[i])
    
    return box_cc            


    
    
    
box,cent,n = load_box() 
#print(box) 
box_clear = []
shan = [4,24,34]
for i in range(len(box)):
    if i in shan:
        continue
    else:
        box_clear.append(box[i])
#del(box[4])
clus = clus(box_clear,0.2)
print(clus)

out = (box_clear,clus)
'''
f = open("result\\boxclus_result.pkl","wb")
pickle.dump(out,f)
f.close()
'''


img = cv2.imread('detec_2.jpg')

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20,20))

C = []
for M in clus:
    C.append(M[0])
for i in range(len(C)):
    x, y, w, h = box_clear[C[i]]
    rect = mpatches.Rectangle((x,y),w,h,fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)
    plt.text(x,y,str(i),color='red')
    
    
'''
for i in clus[2]:
    x, y, w, h = box_clear[i]
    rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)
    plt.text(x,y,str(i),color='red')  
'''
ax.imshow(img)    
plt.show()

    
