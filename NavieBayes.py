#!/usr/bin/env python
#coding:utf-8
import numpy as np


def trainingNaiveBayes(trainingset,traininglabels):
    numTrainDoc = len(trainingset)
    numWords = len(trainingset[0])
    # probability of positive, P(positive)
    p_positive = sum(traininglabels) / float(numTrainDoc)
    
    #count the number of positive and negative
    wordsInPosiNum=np.ones(numWords)
    wordsInNegaNum=np.ones(numWords)
    PosiWordsNum = 2.0
    NegaWordNum = 2.0
    for i in range (0, numTrainDoc):
        if traininglabels[i] ==1: #Positive
            wordsInPosiNum+=trainingset[i]
            PosiWordsNum += sum(trainingset[i])
        else:
            wordsInNegaNum+=trainingset[i]
            NegaWordNum += sum(trainingset[i])
    # calculate P(Wi|P) and P(Wi|N)
    pWordsPositive = np.log(wordsInPosiNum / PosiWordsNum)
    pWordsNegative = np.log(wordsInNegaNum / NegaWordNum)
    
    return pWordsPositive, pWordsNegative, p_positive


def classify(testWordsCount, pWordsPositive, pWordsNegative, p_positive):
    testWordsMarkedArray = np.array(testWordsCount)
    # calculate P(Ci|W), only need to calculate P(W|Ci)*P(Ci)
    p1= sum(testWordsMarkedArray * pWordsPositive) + np.log(p_positive)
    p0= sum(testWordsMarkedArray * pWordsNegative) + np.log(1-p_positive)
    if p1 > p0:
        return 1
    else:
        return 0