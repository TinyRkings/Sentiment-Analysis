#!/usr/bin/env python
#coding:utf-8
from numpy import *


# classify using KNN
def Knn_classfy(newInput, newmarkedwords, dataset,datasetmarkedwords, labels,k):
    squaredDist=[0 for j in range(len(dataset))]
    distance=[0 for j in range(len(dataset))]
    for i in range(len(dataset)):
        for j in range(len(datasetmarkedwords[i])):
            column=datasetmarkedwords[i][j]
            squaredDist[i] +=(dataset[i][column]-newInput[column]) ** 2
        for j in range(len(newmarkedwords)):
            column=newmarkedwords[j]
            squaredDist[i] +=(dataset[i][column]-newInput[column]) ** 2
        distance[i]=squaredDist[i] ** 0.5
        
    ## step 2: sort the distance
    # argsort() returns the indices that would sort an array in a ascending order  
    sortedDistIndices = argsort(distance)
	    
    positive_label=0
    negative_label=0
    predict=0
    for i in range(k):  
        ## step 3: choose the min k distance  
	    voteLabel = labels[sortedDistIndices[i]]
	    ## step 4: count the times labels occur  
	    if(voteLabel==1):
	        positive_label+=1
	    else :
	        negative_label+=1
    ## step 5: the max voted class will return
    if(positive_label>=negative_label):
        predict=1
    else:
        predict=0
    return predict