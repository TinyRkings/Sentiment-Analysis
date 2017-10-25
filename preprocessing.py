#!/usr/bin/env python
#coding:utf-8
import re

def processline(words,stopWords,slangs): # arg words, and internet slangs dictionnary

    processedwords=replaceSlangs(words,slangs).split()
    processedwords2=''  # result variable
    for w in processedwords:
        #strip punctuation
        if w in stopWords:
            None
        else:
            #w=w.replace('''"''', ''' ''')
            processedwords2=processedwords2+w+' '
    return processedwords2.split()
#end


#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
    stopWords = []
    stopWords.append('at_user')
    stopWords.append('url')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

def loadSlangs(filename):
    slangs={}
    fi=open(filename,'r')
    line=fi.readline()
    while line:
        l=line.split(r',%,')
        if len(l) == 2:
            slangs[l[0]]=l[1][:-2]
        line=fi.readline()
    fi.close()
    return slangs

def replaceSlangs(words,slangs):
    result=''
    for w in words:
        if w in slangs.keys():
            result=result+slangs[w]+" "
        else:
            result=result+w+" "
    return result
def process_line(line):
    slangs = loadSlangs('internetSlangs.txt')
    stopWords =getStopWordList('stopWords.txt')
    a=re.findall(r"[\w']+",line)
    b=a[0:len(a)-1]
    
    c=processline(b,stopWords, slangs)
    c.append(a[-1])
    return c
