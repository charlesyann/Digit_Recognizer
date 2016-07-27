#!/usr/bin/pyhon
__author__ = 'lyang'

from numpy import *
import csv

def toInt(array):
	array=mat(array)
	m,n=shape(array)
	newArray=zeros((m,n))
	for i in xrange(m):
		for j in xrange(n):
			newArray[i,j]=int(array[i,j])
	return newArray

def nomalizing(array):
	m,n=shape(array)
	for i in xrange(m):
		for j in xrange(n):
			if array[i,j]!=0:
				array[i,j] = 1
	return array

def loadTrainData():
    l=[]
    with open('../data/train.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line)
    l.remove(l[0])
    l=array(l)
    label=l[:,0]
    data=l[:,1:]
    return nomalizing(toInt(data)),toInt(label)

def loadTestData():
	l=[]
	with open('../data/test.csv') as file:
		lines=csv.reader(file)
		for line in lines:
			l.append(line)
		l.remove(l[0])
		data=array(l)
		return nomalizing(toInt(data))

def loadTestResult():
    l=[]
    with open('knn_benchmark.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line)#28001*2
    l.remove(l[0])
    label=array(l)
    return toInt(label[:,1])  #  label 28000*1


def saveResult(result,csvName):
    with open(csvName,'wb') as myFile:
        myWriter=csv.writer(myFile)
        for i in result:
            tmp=[]
            tmp.append(i)
            myWriter.writerow(tmp)

from sklearn import svm
def svcClassify(trainData,trainLabel,testData):
	# svcClf=svm.SVC(C=5.0)
	svcClf = svm.NuSVC(nu=0.02, gamma=0.02, kernel='rbf', verbose=True)
	svcClf.fit(trainData,ravel(trainLabel))
	testLabel=svcClf.predict(testData)
	saveResult(testLabel,'../result/sklearn_NuSVC_Result.csv')
	return testLabel

trainData,trainLabel=loadTrainData()
testData=loadTestData()
result=svcClassify(trainData,trainLabel,testData)
