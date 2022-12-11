"""
This file contains the code to run the combined neural net and n-gram model. 
"""
import numpy as np
import pandas as pd
import json
from datetime import datetime
from timezonefinder import TimezoneFinder
from pytz import timezone
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim 
import math
import random
import copy
from sklearn.utils import shuffle


torch.manual_seed(0)
NUM_DATA_COLS = 23
numClasses = 2 

FEATURES = ["created_at", "id", "lng", "lat", "topic", "sentiment", "stance", "gender", "temperature_avg", "aggressiveness"]

def extract_tweet_data(source_file):
    """Opens the jsonl file of hydrated tweets to extract the text content and id of the tweet.

    This function sorts through the hydrated tweet objects to extract text content of the
    tweet for further processing. It also saves the id so we can later match the stance. 

    Returns:
        A list of tuples containing the text content of tweet as a string and the tweet id
    """
    tweet_objs = []
    with open(source_file) as f:
        for obj in f:
            tweet_dict = json.loads(obj)
            tweet_objs.append(tweet_dict)
    # list of strings containing the text content of tweets
    text_and_id = [(tweet_objs[i]['data'][j]['text'], int(tweet_objs[i]['data'][j]["id"])) for i in range(len(tweet_objs)) for j in range(len(tweet_objs[i]['data']))]
    tweets = []
    ids = []
    for i in range(len(text_and_id)):
        tweets.append(text_and_id[i][0])
        ids.append(text_and_id[i][1])
    id_stance_df = pd.DataFrame({"id" : ids, "text" : tweets})
    return id_stance_df

'''
Get the labels for all of the tweet ids given.
Must pass in the file where the labels / stances are stored. 
'''
def get_ids(twt_ids, stance_loc):
    data = pd.read_csv(stance_loc)      

    id_dict = data.set_index('id').T.to_dict('list')

    print(len(twt_ids))
    print(len(id_dict))

    twt_labels = []
    for id in twt_ids:
        stance = id_dict[id[0]][0]
        # combine stance of believer and neutral
        if stance == 'believer':
            twt_labels.append(0)
        elif stance == 'neutral':
            twt_labels.append(0)
        else:
            twt_labels.append(1)
    np.savetxt('clara_training_ids_n_gram.csv', twt_labels, fmt='%i')
    return twt_labels

def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Additionally, it removes all stop words (i.e. the most common words that don't
    add to the analysis) and any punctuation.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """
    return message.lower().split(" ")

def create_dictionary(tweets):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    This function is from spam.py from ps2.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python list of frequent words.
    """
    word_dict = {}
    words = []
    for tweet in tweets:
        for word in get_words(tweet[0]):
            if word not in words:
                if word not in word_dict:
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
                    if word_dict[word] >= 5:
                        words.append(word)
    return words 


def create_n_gram(message, n=1):
    """ Generate a list of n-grams from a tweet message.

    Args:
        message: A list of normalized words from the message from get_words.
        n: The number of grams to be generated.

    Returns:
       A list of n-grams from the tweet text.
    """
    temp=zip(*[message[i:] for i in range(0,n)])
    return [" ".join(n_gram) for n_gram in temp]

def create_n_gram_model(tweets, labels, word_dict, n=1):
    """

    Args:
        tweets: A list of strings containing tweet text
        labels: list of labels for tweets
        word_dict: list of strings in our dictionary of relevant words
        n: number of words to be in each n-gram

    Returns:
        A conditional frequency distribution for each n-gram and label
    """
    cfd = ConditionalFreqDist()
    for i, tweet in enumerate(tweets):
        words = [w for w in get_words(tweet[0]) if w in word_dict]
        for gram in create_n_gram(words, n):
            cfd[gram][labels[i]] += 1

    for gram in cfd:
        total = float(sum(cfd[gram].values()))
        for label in cfd[gram]:
            cfd[gram][label] /= total
    
    return cfd

"""
Predict the label
"""      
def predict(model, tweets, word_dict, n=1):
    # sanitize tweets in test set
    final_predictions = []
    for tweet in tweets:
        #denier, neutral, believer
        predictions = [0, 0, 0]
        words = [w for w in get_words(tweet[0]) if w in word_dict]
        for gram in create_n_gram(words, n):
            predictions = [predictions[i] + model[gram][i] for i in range(2)]
        final_predictions.append(np.argmax(predictions))
    return final_predictions


"""
Function to run the n-gram model. Will return
the predictions for the ids which are also being used by
the neural net. Trains on a completely separate set
of rehydrated tweets. 
"""
def main_n_gram():
    # split the tweets into a train and test set
    train_twt = extract_tweet_data('tweet_naive_training.jsonl')
    test_twt = extract_tweet_data('tweet_largeNN_combined.jsonl')

    # split the tweet text and tweet ids to separate lists
    train_twt_text = train_twt[['text']].copy().values.tolist()
    train_twt_ids = train_twt[['id']].copy().values.tolist()

    test_twt_text = test_twt[['text']].copy().values.tolist()
    test_twt_ids = test_twt[['id']].copy().values.tolist()

    test_twt_labels = get_ids(test_twt_ids, 'clara_largeNN_ids_and_stances.csv')       
    train_twt_labels = get_ids(train_twt_ids, 'clara_naive_training_stances.csv')  

    print("Number of train tweets: ", len(train_twt_text))
    print("Number of test tweets: ", len(test_twt_text))

    # create list of words that appear more than 5 times in our training set
    word_dict = create_dictionary(train_twt_text)
    print('Size of dictionary: ', len(word_dict))

    # number of grams
    n = 2
    print("n =", n)
    n_gram_model = create_n_gram_model(train_twt_text, train_twt_labels, word_dict, n)

    predictions = predict(n_gram_model, test_twt_text, word_dict, n)

    n_gram_accuracy = [predictions[i] == test_twt_labels[i] for i in range(len(predictions))].count(True)/len(predictions)
    print('n_gram analysis had an accuracy of {} on the testing set'.format(n_gram_accuracy))
    ids_to_return = []
    for i in range(len(test_twt_ids)):
        ids_to_return.append(str(test_twt_ids[i][0]))
    return predictions, ids_to_return

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

"""
This method collects all the metadata to be used by the model and 
converts it into the proper format. It will also run the n-gram model
in order to generate predicted values for each example, and this
prediction will be used as a model feature. 
"""
def dataImport():
    listOfColsToStack= []
    all_data = pd.read_csv("TwitterDataset.csv")
    data = all_data.sample(n=100000, random_state=2)

    # importing column showing temp change in the location of the tweet 
    tempCol = data[['temperature_avg']]
    tempCol = tempCol.to_numpy()
    tempCol = np.abs(tempCol)

    trueLabels = data[['stance']]
    trueLabels = trueLabels.to_numpy()

    # only use metadata for tweets which were able to be rehydrataed
    preds, ids_used = main_n_gram()
    id_Col = data[['id']]
    id_Col = id_Col.to_numpy()
    rowsToDelete = []

    for index in range(len(tempCol)):
        if math.isnan(tempCol[index][0]):
            rowsToDelete.append(index)
        elif (str(id_Col[index][0]) not in ids_used):
            rowsToDelete.append(index)
    
    # add in n-gram predictions
    preds = np.asarray(preds)
    preds.shape = (preds.shape[0], 1)
    listOfColsToStack.append(preds)

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

    genderCol = data[['gender']]
    genderCol = genderCol.to_numpy()
    genderCol = np.delete(genderCol, rowsToDelete, axis=0)
    femaleCol_oneHot = createOneHotRepresentation(copy.deepcopy(genderCol), "female")
    listOfColsToStack.append(femaleCol_oneHot)
    maleCol_oneHot = createOneHotRepresentation(copy.deepcopy(genderCol), "male")
    listOfColsToStack.append(maleCol_oneHot)

    # creating the sentiment column 
    sentimentCol = data[['sentiment']]
    sentimentCol = sentimentCol.to_numpy()
    sentimentCol = np.delete(sentimentCol, rowsToDelete, axis=0)
    sentimentCol = normVector(sentimentCol)
    listOfColsToStack.append(sentimentCol)
    
    # creating one hot representation for aggressiveness 
    aggressivenessCol = data['aggressiveness']
    aggressivenessCol = aggressivenessCol.to_numpy()
    aggressivenessCol = np.delete(aggressivenessCol, rowsToDelete, axis=0)
    aggressiveness_oneHot = createOneHotRepresentation(aggressivenessCol, "aggressive")
    listOfColsToStack.append(aggressiveness_oneHot)

    topicCol = data[['topic']]
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
    dateCol = data[['created_at']]
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

    """
    Below 20 lines of code is used to convert the 
    UTC timestamp to local timezone. Note that we have already
    removed any tweets without long/lat coordinates. 
    """
    timeCol = data[['created_at']].to_numpy()
    timeCol = np.delete(timeCol, rowsToDelete, axis=0)
    longCol = data[['lng']].to_numpy()
    longCol = np.delete(longCol, rowsToDelete, axis=0)
    latCol = data[['lat']].to_numpy()
    latCol = np.delete(latCol, rowsToDelete, axis=0)

    tf = TimezoneFinder()

    for index in range(len(timeCol)):
        dateStr = str(timeCol[index])
        tz = timezone(tf.timezone_at(lng=longCol[index], lat=latCol[index]))
        date_utc = datetime.fromisoformat(dateStr[2:-2])
        date_local = date_utc.astimezone(tz)
        if date_local.hour <= 4 or date_local.hour >= 20:
            timeCol[index] = 1
        else:
            timeCol[index] = 0

    listOfColsToStack.append(timeCol)

    """
    Uncomment the below lines if you want to generate your own
    set of tweet ids for rehydration. Will also generate a csv file which maps
    the ids being used to their stance aka true label. 
    """
    """
    tweet_ids = data[['id']].to_numpy()
    tweet_ids = np.delete(tweet_ids, rowsToDelete, axis=0)
    tweet_ids = data[['id']].to_numpy()
    tweet_ids = np.delete(tweet_ids, rowsToDelete)
    tweet_ids.tofile('tweet_ids_largeNN.txt', sep='\n')
    tweet_ids.tofile('tweet_ids_combined.txt', sep='\n')
    tweet_ids.shape = (tweet_ids.shape[0], )
    trueLabels.shape = (trueLabels.shape[0], )
    id_stance_df = pd.DataFrame({"id" : tweet_ids, "stance" : trueLabels})
    id_stance_df.to_csv("clara_largeNN_ids_and_stances.csv", index=False)
    """
    

    combinedCols = stackAllCols(listOfColsToStack)
    sampleList = []
    sampleTargets = []

    for index in range(len(combinedCols)):
        currEntry = combinedCols[index]
        currTarget = targetsTensor[index]
        sampleList.append(currEntry)
        sampleTargets.append(currTarget)

    return sampleList, sampleTargets


"""
Split the data into train, eval and test
groups
"""
def splitData(dataset):
    print("dataset length", len(dataset))
    
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
        # dropout here (maybe)
        #self.dropout1 = nn.Dropout(p=.5)
        self.fc3 = nn.Linear(h2Sz, numClasses) # pick output w/ higher number

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.relu1(x) 
        x = self.fc2(x)
        x = self.relu2(x)
        # self.dropout1(x)
        x = self.fc3(x)
        return x 

"""
Train the neural net model
"""
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
            #torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)#added
            optimizer.step() 
            currLoss += loss.item()
        print("Loss for epoch", epoch, "is", currLoss)
           
"""
Evaluate the neural net on the given data and target
labels
"""
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

        # calculate prediction statistics
        return [(numCorrectPreds/len(targets))*100, (numCorrectPreds_neg/numTrueNeg*100), (numCorrectPreds_pos/numPos)*100]

    model.train()

"""
Oversample / undersample in order to create
a dataset more balanced between positive and 
negative examples. 
"""
def oversample_train(samples, targets, rate):
    updatedSamples = []
    updatedTargets = []

    for index in range(len(samples)):
        currEntry = samples[index]
        currTarget = targets[index]
        if currTarget==1: #oversampling positives
            updatedSamples.append(currEntry)
            updatedTargets.append(currTarget)
            if random.randint(1, rate) == 1:
                updatedSamples.append(currEntry)
                updatedTargets.append(currTarget)
        elif random.randint(1, 10) == 1: # undersampling negatives  
            updatedSamples.append(currEntry)
            updatedTargets.append(currTarget)



    updatedSamples, updatedTargets = shuffle(updatedSamples, updatedTargets)

    return updatedSamples, updatedTargets


"""
Evaluate and train the model given model params
"""
def runAndEvalModel(h1Sz, h2Sz, learningRate, numEpochs, sampleList_train, targets_train, sampleList_val, targets_val, sampleList_test, targets_test):
    model = NN(NUM_DATA_COLS, h1Sz, h2Sz)  
    trainModel(model, sampleList_train, targets_train, learningRate, numEpochs) 
    results = {'train': None, 'eval': None, 'test': None}

    results["train"] = evaluateModel(model, sampleList_train, targets_train)

    results["eval"] =  evaluateModel(model, sampleList_val, targets_val)

    results["test"] = evaluateModel(model, sampleList_test, targets_test)

    return results

"""
Function to set up a dataset, including shuffling of
the data. Data will be saved once processed and cleaned. 
"""
def initDataset():
    samples, targets = dataImport()
    print("data loaded")
    print("about to shuffle")
    samples, targets  = shuffle(samples, targets)
    print("shuffle complete")
    dataMapToSave = {"samples": samples, "targets": targets}
    torch.save(dataMapToSave, "climateDatabase_separate_n_gram_time_n2")
    print("data saved ")

"""
Function for running the combined neural net and n-gram model
"""
def main():
    """
    Uncomment the line below to regenerate the dataset from scratch
    instead of loading in specified dataset. 
    """
    #initDataset() 

    rates_to_test = [3]
    '''
    Run the model 10 times with each oversampling rate specified to get 
    aggregate statistics about the accuracy of the model. 
    '''
    for val in rates_to_test:
        overall_train = np.zeros((10, ))
        overall_eval = np.zeros((10, ))
        overall_test = np.zeros((10, ))
        neg_train = np.zeros((10, ))
        neg_eval = np.zeros((10, ))
        neg_test = np.zeros((10, ))
        pos_train = np.zeros((10, ))
        pos_eval = np.zeros((10, ))
        pos_test = np.zeros((10, ))
        for i in range(10):
            print("----------------------------------------------------------------\n")
            print("Oversampling rate is {}".format(val))
            loadedData = torch.load("climateDatabase_separate_n_gram_time_n2")
            samples = loadedData["samples"] 
            targets = loadedData["targets"]

            targets_train, targets_val, targets_test = splitData(targets)
            sampleList_train, sampleList_val, sampleList_test = splitData(samples)

            sampleList_train, targets_train = oversample_train(sampleList_train, targets_train, val) 

            h1Sz=10
            h2Sz=50 
            learningRate= .0015 
            numEpochs=4
            results = runAndEvalModel(h1Sz, h2Sz, learningRate, numEpochs, sampleList_train, targets_train, sampleList_val, targets_val, sampleList_test, targets_test)
            overall_train[i] = results['train'][0]
            overall_eval[i] = results['eval'][0]
            overall_test[i] = results['test'][0]
            pos_train[i] = results['train'][2]
            pos_eval[i] = results['eval'][2]
            pos_test[i] = results['test'][2]
            neg_train[i] = results['train'][1]
            neg_eval[i] = results['eval'][1]
            neg_test[i] = results['test'][1]
        
        '''
        Print out aggregate statistics for the model
        '''
        print("----------------------------------------------------------------\n")
        print("For train, the overall accuracies are: ")
        print("----------------------------------------------------------------\n")
        print("Overall accuracy:", np.mean(overall_train))
        print("Negative accuracy:", np.mean(neg_train))
        print("Positive accuracy:", np.mean(pos_train))
        print("Overall variance:", np.var(overall_train))
        print("Negative variance:", np.var(neg_train))
        print("Positive variance:", np.var(pos_train))

        print("----------------------------------------------------------------\n")
        print("For eval, the overall accuracies are: ")
        print("----------------------------------------------------------------\n")
        print("Overall accuracy:", np.mean(overall_eval))
        print("Negative accuracy:", np.mean(neg_eval))
        print("Positive accuracy:", np.mean(pos_eval))
        print("Overall variance:", np.var(overall_eval))
        print("Negative variance:", np.var(neg_eval))
        print("Positive variance:", np.var(pos_eval))

        print("----------------------------------------------------------------\n")
        print("For test, the overall accuracies are: ")
        print("----------------------------------------------------------------\n")
        print("Overall accuracy:", np.mean(overall_test))
        print("Negative accuracy:", np.mean(neg_test))
        print("Positive accuracy:", np.mean(pos_test))
        print("Overall variance:", np.var(overall_test))
        print("Negative variance:", np.var(neg_test))
        print("Positive variance:", np.var(pos_test))



if __name__ == '__main__':
    main()