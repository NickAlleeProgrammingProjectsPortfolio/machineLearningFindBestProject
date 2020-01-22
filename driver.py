# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 21:56:15 2019

@author: s516583
"""

import pandas as pd
from Find import findTheBestAny
import time
import findHelperTools as fh
from sklearn.model_selection import train_test_split
import glob

#get list of all available datasets to pick from
allDatasets = glob.glob("datasets/*.csv")
print("pick a dataset")
count = 0
for i in allDatasets:
    print(str(count) +": " + i)
    count = count + 1
datasetNumber = int(input("pick the number of the dataset you would like to use."))

#read in the dataset to be used
userDataset = pd.read_csv(allDatasets[datasetNumber])
#clean the dataset
userDataset = fh.removeNonNumbers(userDataset)
allFeatures = fh.grabFeatures(userDataset)
maxFeatures = int(len(allFeatures))-1
count = 0
#print out all feature names to pick from and choice 1 and 2
print("These are the features in the chosen dataset. please review them and either pick one of two choices.")
print("1: pick a feature that you want to see a classifier used on.") 
print("or 2: let the classifier find the particular features that yeild the best result.")
for i in allFeatures:
    print(str(count) + ": " + i)
    count = count + 1

#ask for choice 1 or 2
choice = int(input("pick choice 1 or choice 2 by typing 1 or 2."))
if (choice == 1):
    #ask user which feature to use
    featureChoice = input("which feature would you like? Please type the number of the feature you want.")
    print("you picked: " + str(allFeatures[int(featureChoice)]))
    numberOfXes = len(allFeatures) - 1
if (choice == 2):
    #confirm that they picked choice 2 and ask for an amount of xes to use to determine ys
    print("you picked to let the classifier find the particular features that yeild the best result.")
    numberOfXes = 99999999999999999999999999999999999999999999999999999999999
    choice = 0
    while (numberOfXes > len(allFeatures) - 1):
        numberOfXes = int(input("Enter the max number of features you would like to use (x), to determine a single feature (y). Enter a number equal to or lower than " + str(len(allFeatures) - 1)))

#ask which classifier the user would like to use
classifierChoice = int(input("Please pick the classifier you would like to use. 0: Linear Regression, 1: Binary Classifier, or 2: SVM Classifier. Type 0, 1, or 2."))

#create train and test set
trainSetUserDataset, testSetUserDataset = train_test_split(userDataset, test_size = .25 , random_state = 123)

#find all possibilities with helper function
userDatasetPossibilities = fh.getPossibilities(userDataset,numberOfXes)

#call the functions and report the data while timing it
print("starting calculation")
start = time.perf_counter()
userDatasetClassifier = findTheBestAny(trainSetUserDataset,testSetUserDataset,userDatasetPossibilities,classifierChoice,choice,str(allFeatures[int(featureChoice)]))
stop = time.perf_counter()
amtOfTimeTaken =  stop-start 

print("done in " + str(amtOfTimeTaken) + " seconds.")