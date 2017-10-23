#!/usr/bin/env python
#coding:utf-8
from numpy import *
import sys,os,re 

def mergefiles(in_filenames,merge_name):
    out_file = open(merge_name,"w+")
    for file in in_filenames:
        in_file = open(file,'r')
        out_file.write(in_file.read())
        in_file.close()
    out_file.close()
# parse word in each sentence
def parse_word(merge_name,totalwords,labels):
    out_file = open(merge_name,'r')
    lines=out_file.readlines()
    for line in lines:
        words=re.findall(r"[\w']+",line)
        for index in range(len(words)-1):
            totalwords.append(words[index].lower())
        labels.append(int(words[-1]))
    out_file.close()
    updatewords=list(set(totalwords))
    return updatewords
# calculate the frequency matrix
def cal_frequency(input_file, updatewords,frequency_matrix,size):
    in_file=open(input_file,'r')
    lines=in_file.readlines()
    for line_row in range(len(lines)):
        words=re.findall(r"[\w']+",lines[line_row])
        for index in range(len(words)-1):
            try:
                line_column=updatewords.index(words[index].lower())
                frequency_matrix[line_row+size][line_column]+=1
            except ValueError:
                frequency_matrix[line_row+size][line_column]+=0
    in_file.close()
def write_updatewords(updatewords):
    out_file = open("words.txt","w+")
    out_file.write(str(updatewords))
    out_file.close()
def write_frequency_matrix(frequency_matrix,lines):
    out_file = open("frequency_matrix.txt","w+")
    for line_row in range(lines):
        out_file.write(str(frequency_matrix[line_row]))
        out_file.write('\n')
    out_file.close()
def write_labels(labels):
    out_file = open("labels.txt","w+")
    out_file.write(str(labels))
    out_file.close()
# sum up the frequency of each word
def cal_fre(updatewords, frequency_matrix):
    feature_number=[0 for i in range(len(updatewords))]
    for line_row in range(len(frequency_matrix)):
        for index in range(len(frequency_matrix[0])):
            feature_number[index]+=frequency_matrix[line_row][index]
    return feature_number
# select top k frequency words    
def top_k(feature_numbers, updatewords,k):
    sorted_features=sorted(feature_numbers, reverse= True)
    threshold=sorted_features[k-1]
    prunedwords=[]
    for index in range(len(feature_numbers)):
        if(feature_numbers[index]>= threshold):
            prunedwords.append(updatewords[index])
    return prunedwords
# classify using KNN
def Knn_classfy(newInput, dataset,labels,k):
    numSamples=len(dataset)
    ## step 1: calculate Euclidean distance
    # tile(A, reps): Construct an array by repeating A reps times
    diff = tile(newInput, (numSamples, 1)) - dataset # Subtract element-wise
    squaredDiff = diff ** 2 # squared for the subtract
    squaredDist = sum(squaredDiff, axis = 1) # sum is performed by row
    distance = squaredDist ** 0.5   #list
        
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

def split_data(frequency_matrix, start, size):
    data=[[0 for i in range(len(frequency_matrix[0]))] for j in range(size)]
    for index in range(size):
        for column in range(len(frequency_matrix[0])):
            data[index][column]=frequency_matrix[start+index][column]
    return data
def split_label(labels, start, size):
    newlabels=[]
    for index in range(size):
        newlabels.append(labels[start+index])
    return newlabels                          
if __name__ =='__main__':
    file1="amazon_cells_labelled.txt"
    file2="yelp_labelled.txt"
    file3="imdb_labelled.txt"
    in_filenames=[]
    in_filenames.append(file1)
    in_filenames.append(file2)
    in_filenames.append(file3)
    '\n'.join(in_filenames)
    merge_name="merge.txt"
    mergefiles(in_filenames,merge_name)
    totalwords=[]
    labels=[]
    updatewords=parse_word(merge_name,totalwords,labels)
    frequency_matrix = [[0 for i in range(len(updatewords))] for j in range(len(labels))]
    cal_frequency(file1,updatewords,frequency_matrix,0)
    cal_frequency(file2,updatewords,frequency_matrix,1000)
    cal_frequency(file3,updatewords,frequency_matrix,2000)
    print(len(updatewords))
    
    feature_numbers=cal_fre(updatewords, frequency_matrix)
    topk=1000
    prunedwords=top_k(feature_numbers, updatewords, topk)
    frequency_matrix2=[[0 for i in range(len(prunedwords))] for j in range(len(labels))]
    cal_frequency(file1,prunedwords,frequency_matrix2,0)
    cal_frequency(file2,prunedwords,frequency_matrix2,1000)
    cal_frequency(file3,prunedwords,frequency_matrix2,2000)
    print(len(prunedwords))
    
    
    #split data into training, validation and test set
    start=0
    traingingsize=1800
    validationsize=600
    testsize=600
    trainingset1=split_data(frequency_matrix, start, traingingsize)
    traininglabel1=split_label(labels, start, traingingsize)
    
    validationset1=split_data(frequency_matrix, start+traingingsize, validationsize)
    validationlabel1=split_label(labels, start, validationsize)
    
    tesetset1=split_data(frequency_matrix, start+traingingsize+validationsize, testsize)
    testlabel1=split_label(labels, start, testsize)
    
    trainingset2=split_data(frequency_matrix2, start, traingingsize)
    traininglabel2=traininglabel1
    
    validationset2=split_data(frequency_matrix2, start+traingingsize, validationsize)
    validationlabel2=validationlabel1
    
    tesetset2=split_data(frequency_matrix2, start+traingingsize+validationsize, testsize)
    testlabel2=testlabel1
    
    accuracy=[]
    for k_nearest in range(2):
        matchCount = 0
        for i in range(len(validationset1)):
            predict=Knn_classfy(validationset1[i], trainingset1,traininglabel1,k_nearest+5)
            if(predict==validationlabel1[i]):
                matchCount+=1
        accuracy.append(float(matchCount) / len(validationset1))
    print(accuracy)
    
    accuracy2=[]
    for k_nearest in range(2):
        matchCount = 0
        for i in range(len(validationset2)):
            predict=Knn_classfy(validationset2[i], trainingset2,traininglabel1,k_nearest+5)
            if(predict==validationlabel1[i]):
                matchCount+=1
        accuracy2.append(float(matchCount) / len(validationset2))
    print(accuracy2)
    
    
        
#    write_updatewords(updatewords)
#    write_frequency_matrix(frequency_matrix,len(labels))
#    write_labels(labels)