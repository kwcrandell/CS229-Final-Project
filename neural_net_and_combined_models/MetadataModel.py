"""
This file will run the neural network on metadata only.
Model will run once, and adjustments can be made in 
the main function to change hyperparameters.

Note that you will have to run the initDatabase function
the first time to load in the data. 
"""
import numpy as np
import pandas as pd
from numpy import genfromtxt
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import math
import random
import copy

torch.manual_seed(0)
NUM_DATA_COLS = 21
numClasses = 2 
CSV_PATH = "TwitterDataset.csv"

def normVector(inputVector):
    mean = np.mean(inputVector)
    var = np.std(inputVector)
    return ((inputVector - mean)/var)

def stackAllCols(listOfColsToStack):
    combinedCols = listOfColsToStack[0]
    for index in range(1, len(listOfColsToStack)):
        combinedCols = np.column_stack((combinedCols,listOfColsToStack[index]))
    combinedCols = torch.from_numpy(combinedCols.astype(np.float32))
    print("returning dimensions of ", combinedCols.shape)
    return combinedCols

def createOneHotRepresentation(inputCol, desiredTrait):
    for index in range(len(inputCol)):
        curr = inputCol[index]
        currStr = curr[0] 
        if currStr == desiredTrait:
            inputCol[index] = 1
        else:
            inputCol[index] = 0
    return inputCol

def dataImport():
    listOfColsToStack= []

    # importing column showing temp change in the location of the tweet 
    tempCol = pd.read_csv(CSV_PATH, usecols=[8])
    tempCol = tempCol.to_numpy()
    tempCol = np.abs(tempCol)

    trueLabels = pd.read_csv(CSV_PATH, usecols=[6])
    trueLabels = trueLabels.to_numpy()
    rowsToDelete = []
    for index in range(len(tempCol)):
        if math.isnan(tempCol[index][0]):
            rowsToDelete.append(index)

    tempCol = np.delete(tempCol, rowsToDelete, axis=0)
    tempCol = normVector(tempCol)
    listOfColsToStack.append(tempCol)

    trueLabels = np.delete(trueLabels, rowsToDelete, axis=0)
    targetsList = []
    for index in range(len(trueLabels)):
        if str(trueLabels[index][0]) == "denier": 
            targetsList.append(1) 
        else:
            targetsList.append(0)
    targetsTensor = torch.LongTensor(targetsList)

    genderCol = pd.read_csv(CSV_PATH, usecols=[7])
    genderCol = genderCol.to_numpy()
    genderCol = np.delete(genderCol, rowsToDelete, axis=0)
    femaleCol_oneHot = createOneHotRepresentation(copy.deepcopy(genderCol), "female")
    listOfColsToStack.append(femaleCol_oneHot)
    maleCol_oneHot = createOneHotRepresentation(copy.deepcopy(genderCol), "male")
    listOfColsToStack.append(maleCol_oneHot)

    # creating the sentiment column 
    sentimentCol = pd.read_csv(CSV_PATH, usecols=[5])
    sentimentCol = sentimentCol.to_numpy()
    sentimentCol = np.delete(sentimentCol, rowsToDelete, axis=0)
    sentimentCol = normVector(sentimentCol)
    listOfColsToStack.append(sentimentCol)
    
    # creating one hot representation for aggressiveness 
    aggressivenessCol = pd.read_csv(CSV_PATH, usecols=[9])
    aggressivenessCol = aggressivenessCol.to_numpy()
    aggressivenessCol = np.delete(aggressivenessCol, rowsToDelete, axis=0)
    aggressiveness_oneHot = createOneHotRepresentation(aggressivenessCol, "aggressive")
    listOfColsToStack.append(aggressiveness_oneHot)

    topicCol = pd.read_csv(CSV_PATH, usecols=[4])
    topicCol = topicCol.to_numpy()
    topicCol = np.delete(topicCol, rowsToDelete, axis=0)

    topic1_oneHot = createOneHotRepresentation(copy.deepcopy(topicCol), "Ideological Positions on Global Warming")
    listOfColsToStack.append(topic1_oneHot)
    topic2_oneHot = createOneHotRepresentation(copy.deepcopy(topicCol), "Impact of Resource Overconsumption")
    listOfColsToStack.append(topic2_oneHot)
    topic3_oneHot = createOneHotRepresentation(copy.deepcopy(topicCol), "Global stance")
    listOfColsToStack.append(topic3_oneHot)
    topic4_oneHot = createOneHotRepresentation(copy.deepcopy(topicCol), "Weather Extremes")
    listOfColsToStack.append(topic4_oneHot)
    topic5_oneHot = createOneHotRepresentation(copy.deepcopy(topicCol), "Undefined / One Word Hashtags")
    listOfColsToStack.append(topic5_oneHot)
    topic6_oneHot = createOneHotRepresentation(copy.deepcopy(topicCol), "Seriousness of Gas Emissions")
    listOfColsToStack.append(topic6_oneHot)
    topic7_oneHot = createOneHotRepresentation(copy.deepcopy(topicCol), "Importance of Human Intervantion")
    listOfColsToStack.append(topic7_oneHot)
    topic8_oneHot = createOneHotRepresentation(copy.deepcopy(topicCol), "Donald Trump versus Science")
    listOfColsToStack.append(topic8_oneHot)
    topic9_oneHot = createOneHotRepresentation(copy.deepcopy(topicCol), "Significance of Pollution Awareness Events")
    listOfColsToStack.append(topic9_oneHot)
    topic10_oneHot = createOneHotRepresentation(copy.deepcopy(topicCol), "Politics")    
    listOfColsToStack.append(topic10_oneHot)

    # creating column that has the year 
    dateCol = pd.read_csv(CSV_PATH, usecols=[0])
    dateCol = dateCol.to_numpy()
    for index in range(len(dateCol)):
        oldStr = str(dateCol[index])
        dateCol[index] = oldStr[2:6]

    dateCol = np.delete(dateCol, rowsToDelete, axis=0)
    date2006_oneHot = createOneHotRepresentation(copy.deepcopy(dateCol), "2006")    
    listOfColsToStack.append(date2006_oneHot)
    date2007_oneHot = createOneHotRepresentation(copy.deepcopy(dateCol), "2007")    
    listOfColsToStack.append(date2007_oneHot)
    date2008_oneHot = createOneHotRepresentation(copy.deepcopy(dateCol), "2008")    
    listOfColsToStack.append(date2008_oneHot) 
    date2009_oneHot = createOneHotRepresentation(copy.deepcopy(dateCol), "2009")    
    listOfColsToStack.append(date2009_oneHot)
    date2010_oneHot = createOneHotRepresentation(copy.deepcopy(dateCol), "2010")    
    listOfColsToStack.append(date2010_oneHot)
    date2011_oneHot = createOneHotRepresentation(copy.deepcopy(dateCol), "2011")    
    listOfColsToStack.append(date2011_oneHot)

    combinedCols = stackAllCols(listOfColsToStack)
    sampleList = []
    sampleTargets = []

    for index in range(len(combinedCols)):
        currEntry = combinedCols[index]
        currTarget = targetsTensor[index]
        sampleList.append(currEntry)
        sampleTargets.append(currTarget)

    return sampleList, sampleTargets


def splitData(dataset):    
    endTrain = int(len(dataset)*.84)
    endVal = int(endTrain + len(dataset)*.08)
    dataset_train = dataset[:endTrain]
    dataset_val = dataset[endTrain:endVal]
    dataset_test = dataset[endVal:]

    return dataset_train, dataset_val, dataset_test

class NN(nn.Module):
    def __init__(self, inputSz, h1Sz, h2Sz, numClasses=2):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(inputSz, h1Sz)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(h1Sz, h2Sz)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(h2Sz, numClasses) 

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.relu1(x) 
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x 

def trainModel(model, data, targets, learningRate, numEpochs):
    lossFn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learningRate)
    for epoch in range(numEpochs):
        data, targets = shuffle(data, targets)
        currLoss = 0.0
        for index in range(len(targets)):
            target=targets[index]
            outputs = model(data[index])
            loss = lossFn(outputs, targets[index])
            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step() 
            currLoss += loss.item()
        print("Loss for epoch", epoch, "is", currLoss)
           

def evaluateModel(model, data, targets):
    model.eval()
    with torch.no_grad():
        index=0
        numCorrectPreds=0
        negPred=0
        numCorrectPreds_pos = 0 
        predictions = []
        for currTensor in data:
            prediction = model(currTensor)
            predicted_class = np.argmax(prediction)
            predictions.append(predicted_class.item())
            if predicted_class.item() == targets[index].item():
                numCorrectPreds += 1 
                if predicted_class.item() == 1:
                        numCorrectPreds_pos += 1
            if predicted_class.item() == 0:
                negPred+=1
            index+=1
        numCorrectPreds_neg = numCorrectPreds - numCorrectPreds_pos
        
        numPos = 0 
        for curr in targets:
            if curr==1:
                numPos+=1
        numTrueNeg = len(targets)-numPos
        

        print("Percent of positive samples in the dataset: ", (numPos/len(targets))*100)
        print("Percent of negative samples in the dataset:", (numTrueNeg/len(targets)*100))
        print("----------------------------------------------------------------")
        print("Overall accuracy:", (numCorrectPreds/len(targets))*100)
        print("Negative accuracy:", (numCorrectPreds_neg/numTrueNeg*100))
        print("Positive accuracy:", (numCorrectPreds_pos/numPos)*100)
        overallAccuracy = (numCorrectPreds/len(targets))*100
        negAccuracy = (numCorrectPreds_neg/numTrueNeg*100) 
        posAccuracy = (numCorrectPreds_pos/numPos)*100


    model.train()
    return overallAccuracy, negAccuracy, posAccuracy

def oversample_train(samples, targets, rate):
    updatedSamples = []
    updatedTargets = []

    for index in range(len(samples)):
        currEntry = samples[index]
        currTarget = targets[index]
        if currTarget==1: #oversampling positives
            updatedSamples.append(currEntry)
            updatedTargets.append(currTarget)
            if random.randint(1, 10) == 1:
                updatedSamples.append(currEntry)
                updatedTargets.append(currTarget)
        elif random.randint(1, 10) == 1: # undersampling negatives
            updatedSamples.append(currEntry)
            updatedTargets.append(currTarget)

    updatedSamples, updatedTargets = shuffle(updatedSamples, updatedTargets)

    return updatedSamples, updatedTargets

from sklearn.utils import shuffle


def runAndEvalModel(h1Sz, h2Sz, learningRate, numEpochs, sampleList_train, targets_train, sampleList_val, targets_val, sampleList_test, targets_test):
    model = NN(NUM_DATA_COLS, h1Sz, h2Sz)  
    print("Training the model")
    trainModel(model, sampleList_train, targets_train, learningRate, numEpochs) 

    print("Evaluating the model on the training set")
    overallAccuracy, negAccuracy, posAccuracy = evaluateModel(model, sampleList_train, targets_train)

    print("Evaluating the model on the validation set")
    overallAccuracyVal, negAccuracyVal, posAccuracyVal = evaluateModel(model, sampleList_val, targets_val)
    

    print("Evaluating the model on the test set")
    evaluateModel(model, sampleList_test, targets_test)
    return overallAccuracyVal, negAccuracyVal, posAccuracyVal 

def initDataset():
    samples, targets = dataImport()
    print("data loaded")
    print("about to shuffle")
    samples, targets  = shuffle(samples, targets)
    print("shuffle complete")
    dataMapToSave = {"samples": samples, "targets": targets}
    torch.save(dataMapToSave, "climateDatabase") 
    print("data saved ")

import matplotlib.pyplot as plt

def main():    
    h1Sz=10
    h2Sz=50 
    learningRate= .0015 
    numEpochs=4
    xCoor = []
    yCoorOverall = []
    yCoorNeg = []
    yCoorPos = []
    currRate = 10 

    initDataset() 
    loadedData = torch.load("climateDatabase")
    samples = loadedData["samples"] 
    targets = loadedData["targets"]
    print("Climate database loaded")

    targets_train, targets_val, targets_test = splitData(targets)
    sampleList_train, sampleList_val, sampleList_test = splitData(samples)
    sampleList_train, targets_train = oversample_train(sampleList_train, targets_train, currRate) 
    overallAccuracyVal, negAccuracyVal, posAccuracyVal = runAndEvalModel(h1Sz, h2Sz, learningRate, numEpochs, sampleList_train, targets_train, sampleList_val, targets_val, sampleList_test, targets_test)
    

    xCoor = []
    yCoorOverall = []
    yCoorNeg = []
    yCoorPos = []

    '''
    # Random hyperparameter tuning 
    h1Sz=random.randint(5, 250)
    h2Sz=random.randint(5, 250)
    learningRate= random.uniform(.001,.1)
    numEpochs=random.randint(5, 30)

    runAndEvalModel(h1Sz, h2Sz, learningRate, numEpochs, sampleList_train, targets_train, sampleList_val, targets_val, [], [])
    '''

    '''

    # graphs for the hidden layer sizes  
    #sampleList_train, targets_train = oversample_train(sampleList_train, targets_train, rate) 
    
    currHiddenLayerSz = 0
    for it in range(1,10):
        currHiddenLayerSz += 10
        print("Size of  hidden layer", currHiddenLayerSz)
        xCoor.append(currHiddenLayerSz)

        overallAccuracyVal, negAccuracyVal, posAccuracyVal = runAndEvalModel(h1Sz, currHiddenLayerSz, learningRate, numEpochs, sampleList_train, targets_train, sampleList_val, targets_val, sampleList_test, targets_test)
       
        yCoorOverall.append(overallAccuracyVal)
        yCoorNeg.append(negAccuracyVal)
        yCoorPos.append(posAccuracyVal)
    plt.xlabel("Size of Second Hidden Layer")
    plt.ylabel("Accuracy")
    plt.plot(xCoor, yCoorOverall)
    plt.plot(xCoor, yCoorNeg)
    plt.plot(xCoor, yCoorPos)
    plt.legend(["Overall accuracy", "Negative accuracy", "Positive accuracy"])
    plt.savefig('hidden_2.png')
    print("Hidden 2 Sz graph saved")
    

    
    # graph for the learning rate 
    for it in range(1,15):

        LR = it*.0005
        print("Learning rate", LR)
        xCoor.append(LR)

        overallAccuracyVal, negAccuracyVal, posAccuracyVal = runAndEvalModel(h1Sz, h2Sz, LR, numEpochs, sampleList_train, targets_train, sampleList_val, targets_val, sampleList_test, targets_test)
        yCoorOverall.append(overallAccuracyVal)
        yCoorNeg.append(negAccuracyVal)
        yCoorPos.append(posAccuracyVal)
    plt.plot(xCoor, yCoorOverall)
    plt.plot(xCoor, yCoorNeg)
    plt.plot(xCoor, yCoorPos)
    plt.legend(["Overall accuracy", "Negative accuracy", "Positive accuracy"])
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.savefig('learning_rate_graph.png')
    print("Learning rate graph saved ")
    '''

    '''
    # graph for the number of epochs 
    for epochNum in range(2,11):
        print("Epoch number", epochNum)
        xCoor.append(epochNum)

        overallAccuracyVal, negAccuracyVal, posAccuracyVal = runAndEvalModel(h1Sz, h2Sz, learningRate, epochNum, sampleList_train, targets_train, sampleList_val, targets_val, sampleList_test, targets_test)
       
        yCoorOverall.append(overallAccuracyVal)
        yCoorNeg.append(negAccuracyVal)
        yCoorPos.append(posAccuracyVal)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.plot(xCoor, yCoorOverall)
    plt.plot(xCoor, yCoorNeg)
    plt.plot(xCoor, yCoorPos)
    plt.legend(["Overall accuracy", "Negative accuracy", "Positive accuracy"])
    plt.savefig('number_of_epochs.png')
    print("Epoch graph saved")
    '''


if __name__ == '__main__':
    main()
