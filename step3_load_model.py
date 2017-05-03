# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 09:46:19 2017

@author: admin
"""

from keras.models import load_model
import pickle
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

def predict_boxes(boxes):
    '''
    Use pre-trained CNN model to predict a best box in a cluster.
    '''
    n = len(boxes)
    img = cv2.imread('detec_2.jpg',0)
    dataX = np.zeros((n,74,74,1))
    
    for i in range(n):
        x,y,w,h = boxes[i]
        box_img = img[y:y+h,x:x+w]
        box_res = cv2.resize(box_img,(74,74),interpolation=cv2.INTER_CUBIC)
        dataX[i] = box_res.reshape(74,74,1)
    
    model = load_model('my_model.h5')    
    dataY = model.predict(dataX)
    
    predict = np.zeros((len(dataY),2))
    for t in range(len(dataY)):
        predict[t,0] = np.argmax(dataY[t])
        predict[t,1] = np.max(dataY[t])
       
    return dataX,dataY,predict

def real_test(clus,predict):
    ground_truth = [0,2,3,2,1,0,0,0,4,4,2,1,3,1,2]
    accuracy = 0
    tagset = np.zeros(38,dtype="int64")
    for i in range(len(clus)):
        for j in clus[i]:
            tagset[j] = ground_truth[i]
            if predict[j][0] == ground_truth[i]:
                accuracy += 1
    return accuracy ,tagset

    
def best_pre(box,cluster,dataY,predict):
    All = []
    
    for i in range(len(cluster)):
        p = [predict[x][1] for x in cluster[i]]
        parg = np.argmax(p)
        box_id = cluster[i][parg]
        All.append([box_id,predict[box_id][0],predict[box_id][1]])
        

    return All   

def draw(box,All,P):
    #box_id, label, pb = All[i]
    img = cv2.imread('detec_2.jpg')
    lis = ["nut","flange","screw","eyelet","bracket"]
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,10))
    
    for T in All:        
        x, y, w, h = box[T[0]]
        rect = mpatches.Rectangle((x,y),w,h,fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        plt.text(x,y,lis[int(T[1])],color='red')     
        
    '''
    for i in range(len(box)):
        x, y, w, h = box[i]
        rect = mpatches.Rectangle((x,y),w,h,fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        label = lis[int(P[i][0])]
        plt.text(x,y,str(i)+' '+label+str(P[i][1]),color='red')       
    '''    
    
    ax.imshow(img)
    plt.show()
    
    
def step3_main():
    
    ff = open("result\\boxclus_result.pkl","rb")
    box, clus = pickle.load(ff)
    ff.close()
    
    X, Y, P = predict_boxes(box)
    #print(Y,P)
    #print(len(Y))
    
    accuracy,tag = real_test(clus,P)
    print(accuracy,'/ 41')
    
    #print(np.shape(X))
    #print(tag)
    Al = best_pre(box,clus,Y,P)
    #print(Al)
    draw(box,Al,P)
    return X,tag

if __name__ == '__main__':
    X,tag = step3_main()
    
    
    
