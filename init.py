#!/usr/bin/env python
#coding:utf-8
import numpy as np
import sys,os,re
import NavieBayes as naiveBayes
import knn as knn
import preprocessing
import time 

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
        #words=re.findall(r"[\w']+",line)
        words=preprocessing.process_line(line)
        for index in range(len(words)-1):
            totalwords.append(words[index].lower())
        labels.append(int(words[-1]))
    out_file.close()
    updatewords=list(set(totalwords))
    return updatewords
# calculate the frequency matrix
def cal_frequency(input_file, updatewords,frequency_matrix,updatemarkedwords):
    in_file=open(input_file,'r')
    lines=in_file.readlines()
    for line_row in range(len(lines)):
        #words=re.findall(r"[\w']+",lines[line_row])
        words=preprocessing.process_line(lines[line_row])
        for index in range(len(words)-1):
            try:
                line_column=updatewords.index(words[index].lower())
                frequency_matrix[line_row][line_column]+=1
                updatemarkedwords[line_row].append(line_column)
            except ValueError:
                pass
        updatemarkedwords[line_row]=list(set(updatemarkedwords[line_row]))        
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
    
def split_data(frequency_matrix, start, size):
    data = frequency_matrix[start:start+size]
    return data
def split_label(labels, start, size):
    newlabels=labels[start:start+size]
    return newlabels
def split_marked(updatemarkedwords, start, size):
    newmarked=updatemarkedwords[start:start+size]
    return newmarked   
                    
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
    #mergefiles(in_filenames,merge_name)
    f1=open(file1,'r')
    f2=open(file2,'r')
    f3=open(file3,'r')
    f4=open(merge_name,"w+")
    for i in range(1000):
        f4.write(f1.readline())
        f4.write(f2.readline())
        f4.write(f3.readline())
    f4.close()
    f1.close()
    f2.close()
    f3.close()
    totalwords=[]
    labels=[]
    updatewords=parse_word(merge_name,totalwords,labels)
    frequency_matrix = [[0 for i in range(len(updatewords))] for j in range(len(labels))]
    updatemarkedwords= [[0 for i in range(1)] for j in range(len(labels))]
    cal_frequency(merge_name, updatewords, frequency_matrix,updatemarkedwords)
    print "original feature size: ", len(updatewords)
    
    feature_numbers=cal_fre(updatewords, frequency_matrix)
    topk=1000
    prunedwords=top_k(feature_numbers, updatewords, topk)
    frequency_matrix2=[[0 for i in range(len(prunedwords))] for j in range(len(labels))]
    updatemarkedwords2= [[0 for i in range(1)] for j in range(len(labels))]
    cal_frequency(merge_name, prunedwords, frequency_matrix2,updatemarkedwords2)
    print "pruned feature size: ", len(prunedwords)
    
    
    ##split data into training, validation and test set
    start=0
    trainingsize=2100
    validationsize=450
    testsize=450
    trainingset1=split_data(frequency_matrix, start, trainingsize)
    traininglabel1=split_label(labels, start, trainingsize)
    trainingmarkedwords1=split_marked(updatemarkedwords, start, trainingsize)
    
    validationset1=split_data(frequency_matrix, start+trainingsize, validationsize)
    validationlabel1=split_label(labels, start+trainingsize, validationsize)
    validationmarkedwords1=split_marked(updatemarkedwords, start+trainingsize, validationsize)
    
    testset1=split_data(frequency_matrix, start+trainingsize+validationsize, testsize)
    testlabel1=split_label(labels, start+trainingsize+validationsize, testsize)
    testmarkedwords1=split_marked(updatemarkedwords, start+trainingsize+validationsize, testsize)
    
    trainingset2=split_data(frequency_matrix2, start, trainingsize)
    traininglabel2=traininglabel1
    trainingmarkedwords2=split_marked(updatemarkedwords2, start, trainingsize)
    
    validationset2=split_data(frequency_matrix2, start+trainingsize, validationsize)
    validationlabel2=validationlabel1
    validationmarkedwords2=split_marked(updatemarkedwords2, start+trainingsize, validationsize)
    
    testset2=split_data(frequency_matrix2, start+trainingsize+validationsize, testsize)
    testlabel2=testlabel1
    testmarkedwords2=split_marked(updatemarkedwords2, start+trainingsize+validationsize, testsize)
    
    ##KNN classifier
    #choose the k which has biggest accuracy in validation set
    print '\n'
    print "Build original KNN by choosing the best k_nearest"
    print "---------------------------------------"
    k_initial=1
    k_size=20 
    accuracy=[]
    start_time = time.time()
    for k_nearest in range(k_size):
        matchCount = 0
        for i in range(len(validationset1)):
            predict=knn.Knn_classfy(validationset1[i], validationmarkedwords1[i], trainingset1,trainingmarkedwords1, traininglabel1,k_nearest+k_initial)
            if(predict==validationlabel1[i]):
                matchCount+=1
        accuracy.append(float(matchCount) / len(validationset1))
    end_time= time.time() - start_time
    select_k1=accuracy.index(max(accuracy))
    choose_k1=select_k1+k_initial
    print "The time to build original KNN by choosing the best k_nearest: ", end_time, "s" 
    print "k size for original feature set:", choose_k1
    print "accuracy for original validation feature set in KNN:", accuracy[select_k1]
    
    #test the accuracy on testing sets
    print '\n'
    print "Test the accuracy on original testing set in KNN"
    print "---------------------------------------"    
    matchCount=0
    record_time=[]
    for i in range(len(testset1)):
        start_ti = time.time()
        predict=knn.Knn_classfy(testset1[i], testmarkedwords1[i], trainingset1, trainingmarkedwords1, traininglabel1,choose_k1)
        end_ti= time.time() - start_ti
        record_time.append(end_ti)
        if(predict==testlabel1[i]):
            matchCount+=1
    accuracy_test=float(matchCount)/len(testset1)
    print "time to classify a new original tuple in KNN:", sum(record_time)/len(record_time) 
    print "accuracy for original test feature set in KNN:", accuracy_test  
    
    #test the accuracy on training sets
    print '\n'
    print "Test the accuracy on original training set in KNN"
    print "---------------------------------------" 
    matchCount=0
    for i in range(len(trainingset1)):
        predict=knn.Knn_classfy(trainingset1[i], trainingmarkedwords1[i], trainingset1, trainingmarkedwords1, traininglabel1,choose_k1)
        if(predict==traininglabel1[i]):
            matchCount+=1
    accuracy_test=float(matchCount)/len(trainingset1)
    print "accuracy for original training feature set in KNN:", accuracy_test
    
    print '\n'
    print "Build pruned KNN by choosing the best k_nearest"
    print "---------------------------------------"
    accuracy2=[]
    start_time2 = time.time()
    for k_nearest in range(k_size):
        matchCount = 0
        for i in range(len(validationset2)):
            predict=knn.Knn_classfy(validationset2[i], validationmarkedwords2[i], trainingset2,trainingmarkedwords2, traininglabel2,k_nearest+k_initial)
            if(predict==validationlabel2[i]):
                matchCount+=1
        accuracy2.append(float(matchCount) / len(validationset2))
    end_time2= time.time() - start_time2
    select_k2=accuracy2.index(max(accuracy2))
    choose_k2=select_k2+k_initial
    print "The time to build pruned KNN by choosing the best k_nearest: ", end_time2, "s"
    print "k size for pruned feature set:", choose_k2
    print "accuracy for pruned validation feature set in KNN:", accuracy2[select_k2]
      
    #test the accuracy on testing sets
    print '\n'
    print "Test the accuracy on pruned testing set in KNN"
    print "---------------------------------------"
    matchCount2=0
    record_time2=[]
    for i in range(len(testset2)):
        start_ti = time.time()
        predict=knn.Knn_classfy(testset2[i], testmarkedwords2[i], trainingset2, trainingmarkedwords2, traininglabel2,choose_k2)
        end_ti= time.time() - start_ti
        record_time2.append(end_ti)
        if(predict==testlabel2[i]):
            matchCount2+=1
    accuracy_test2=float(matchCount2)/len(testset2)
    print "time to classify a new pruned tuple in KNN:", sum(record_time2)/len(record_time2) 
    print "accuracy for pruned test feature set in KNN:", accuracy_test2
    
    #test the accuracy on training sets
    print '\n'
    print "Test the accuracy on pruned training set in KNN"
    print "---------------------------------------" 
    matchCount2=0
    for i in range(len(trainingset2)):
        predict=knn.Knn_classfy(trainingset2[i], trainingmarkedwords2[i], trainingset2, trainingmarkedwords2, traininglabel2,choose_k2)
        if(predict==traininglabel2[i]):
            matchCount2+=1
    accuracy_test2=float(matchCount2)/len(trainingset2)
    print "accuracy for pruned training feature set in KNN:", accuracy_test2
    
    ##Naive Bayes Classifier
    #train the model
    print '\n'
    print "Build Original Naive Bayes Classifer"
    print "---------------------------------------"
    start_time3 = time.time()
    pWordsPositive, pWordsNegative, p_positive = naiveBayes.trainingNaiveBayes(trainingset1, traininglabel1)
    end_time3= time.time() - start_time3
    print "The time to build original Naive Bayes Classifier: ", end_time3, "s"
    
    #test the model
    print '\n'
    print "Test the accuracy on original testing set in Naive Bayes"
    print "---------------------------------------"
    correctCount =0.0
    record_time3=[]
    for i in range (len(testset1)):
        start_ti = time.time()
        predict_type = naiveBayes.classify(testset1[i],pWordsPositive, pWordsNegative, p_positive)
        end_ti= time.time() - start_ti
        record_time3.append(end_ti) 
        if(predict_type == testlabel1[i]):
            correctCount +=1
    accuracy_bayes1=float(correctCount)/len(testset1)
    print "time to classify a new original tuple in Naive Bayes:", sum(record_time3)/len(record_time3) 
    print "accuracy for original testing set in Naive Bayes Classifier: ", accuracy_bayes1
    
    print '\n'
    print "Test the accuracy on original validation set in Naive Bayes"
    print "---------------------------------------"
    correctCount =0.0
    for i in range (len(validationset1)):
        predict_type = naiveBayes.classify(validationset1[i],pWordsPositive, pWordsNegative, p_positive)
        if(predict_type == validationlabel1[i]):
            correctCount +=1
    accuracy_bayes1=float(correctCount)/len(validationset1)  
    print "accuracy for original validation set in Naive Bayes Classifier: ", accuracy_bayes1
    
    print '\n'
    print "Test the accuracy on original training set in Naive Bayes"
    print "---------------------------------------"
    correctCount =0.0
    for i in range (len(trainingset1)):
        predict_type = naiveBayes.classify(trainingset1[i],pWordsPositive, pWordsNegative, p_positive)
        if(predict_type == traininglabel1[i]):
            correctCount +=1
    accuracy_bayes1=float(correctCount)/len(trainingset1)  
    print "accuracy for original training set in Naive Bayes Classifier: ", accuracy_bayes1
    
    
    
    #train the model
    print '\n'
    print "Build Pruned Naive Bayes Classifer"
    print "---------------------------------------"
    start_time4 = time.time()
    pWordsPositive2, pWordsNegative2, p_positive2 = naiveBayes.trainingNaiveBayes(trainingset2, traininglabel2)
    end_time4= time.time() - start_time4
    print "The time to build pruned Naive Bayes Classifier: ", end_time4, "s"
    
    #test the model
    print '\n'
    print "Test the accuracy on pruned testing set in Naive Bayes"
    print "---------------------------------------"
    correctCount2 =0.0
    record_time4=[]
    for i in range (len(testset2)):
        start_ti = time.time()
        predict_type = naiveBayes.classify(testset2[i],pWordsPositive2, pWordsNegative2, p_positive2)
        end_ti= time.time() - start_ti
        record_time4.append(end_ti) 
        if(predict_type == testlabel2[i]):
            correctCount2 +=1
    accuracy_bayes2=float(correctCount2)/len(testset2)
    print "time to classify a new pruned tuple in Naive Bayes:", sum(record_time4)/len(record_time4) 
    print "accuracy for pruned testing set in Naive Bayes Classifier: ", accuracy_bayes2
    
    print '\n'
    print "Test the accuracy on pruned validation set in Naive Bayes"
    print "---------------------------------------"
    correctCount2 =0.0
    for i in range (len(validationset2)):
        predict_type = naiveBayes.classify(validationset2[i],pWordsPositive2, pWordsNegative2, p_positive2)
        if(predict_type == validationlabel2[i]):
            correctCount2 +=1
    accuracy_bayes2=float(correctCount2)/len(validationset2)
    print "accuracy for pruned validation set in Naive Bayes Classifier: ", accuracy_bayes2
    
    print '\n'
    print "Test the accuracy on pruned training set in Naive Bayes"
    print "---------------------------------------"
    correctCount2 =0.0
    for i in range (len(trainingset2)):
        predict_type = naiveBayes.classify(trainingset2[i],pWordsPositive2, pWordsNegative2, p_positive2)
        if(predict_type == traininglabel2[i]):
            correctCount2 +=1
    accuracy_bayes2=float(correctCount2)/len(trainingset2)
    print "accuracy for pruned training set in Naive Bayes Classifier: ", accuracy_bayes2
    
    
        
#    write_updatewords(updatewords)
#    write_frequency_matrix(frequency_matrix,len(labels))
#    write_labels(labels)