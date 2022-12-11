"""
Will run all three language models.

This will run for 5000 tweets, 
naive bayes and n_gram can also be updated
to run on the 80000 tweet dataset found in 
the neural_net_and_combined_models folder,
which will take around an hour to run. 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import os
import tqdm
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.probability import ConditionalFreqDist
from gensim.models import Word2Vec



def extract_tweet_data():
    """Opens the jsonl file of hydrated tweets to extract the text content and id of the tweet.

    This function sorts through the hydrated tweet objects to extract text content of the
    tweet for further processing. It also saves the id so we can later match the stance. 

    This list of [tweet, id] pairs is converted to a Pandas Dataframe to be saved to a csv.

    Returns:
        A  Dataframe containing the text content of tweet as a string and the tweet id
    """
    tweet_objs = []
    filename = '5000_tweets.jsonl'
    with tqdm.tqdm(total=os.path.getsize(filename)) as pbar:
        with open(filename) as f:
            for obj in f:
                pbar.update(len(obj))
                tweet_dict = json.loads(obj)
                tweet_objs.append(tweet_dict)
    # list of strings containing the text content of tweets and the related ID
    text_and_id = [[tweet_objs[i]['data'][j]['text'], int(tweet_objs[i]['data'][j]["id"])] for i in range(len(tweet_objs)) for j in range(len(tweet_objs[i]['data']))]
    ab = pd.DataFrame(np.array(text_and_id), columns=['text', 'id'])
    ab['id'] = pd.to_numeric(ab['id'])
    ab.to_csv('5000_tweets_and_id.csv')
    return ab

def get_stances():
    """ Retrieves the ids and stances from the Kaggle Twitter dataset.

    This function reads the csv file of the Twitter Dataset, extracts the id and stance columns,
    saves the subset of data for future use, and returns the dataframe.

     Returns:
        A Dataframe containing the stance and id of all tweets in the dataset.

    """
    data = pd.read_csv('TwitterDataset.csv')
    stance_and_id = data[["id", "stance"]]
    stance_and_id.to_csv('all_stance_and_id.csv', index=False)
    return stance_and_id

def balance_data(df):
    """ Makes the counts of all stances in the dataset equal.
    """
    count = len(df[df['stance'] == 'denier'])

    balanced_df = pd.concat(
        [df[df['stance'] == 'believer'].sample(n=int(count)),
        df[df['stance'] == 'neutral'].sample(n=int(count)),
        df[df['stance'] == 'denier'].sample(n=count)]
    )
    return balanced_df

def get_tweets_and_labels():
    """ Creates the train and test datasets. 

    This function reads in the ids, stances, and tweets of the dataset or calls
    get_stances() and extract_tweet_data() to get those. 

    Since the dataset contains mainly positive tweets, we included code to randomly
    sample equal numbers of each class. We did not end up using this sampling for 
    the final analysis.

    After the preprocessing is done, the train and test dataframes are saved for
    future reference.

    Returns:
        train_df: dataframe containing the training tweets, ids, and labels.
        test_df: dataframe containing the testing tweets, ids, and labels.
    """
    #ids_and_stances = get_stances()
    ids_and_stances = pd.read_csv('5000_ids_and_stances.csv')
    #twts_and_ids = extract_tweet_data()
    twts_and_ids = pd.read_csv('5000_tweets_and_ids.csv')

    df = twts_and_ids.merge(ids_and_stances, on='id')

    # uncomment if you want a balanced dataset of tweet stances
    #df = balance_data(df)

    mapping = {'believer': 1, 'neutral': 1, 'denier': 0}
    balanced_df = df.replace({'stance': mapping})
    balanced_df = balanced_df.rename(columns={'stance': 'label'})

    # split the tweets into a train and test set
    train_df, test_df = train_test_split(balanced_df, test_size=0.2)
  
    train_df = train_df.reset_index()
    test_df = test_df.reset_index()

    train_df = train_df.drop(columns=['index'])
    test_df = test_df.drop(columns=['index'])

    train_df.to_csv('train_df.csv', index=False)
    test_df.to_csv('test_df.csv', index=False)

    return train_df, test_df

def create_dictionary(tokens_list):
    """Create a dataframe mapping words to integer indices.

    This function creates a dataframe dictionary of words to indices.

    Rare words are often not useful for modeling, so we only included words
    they occured in more than 5 messages.

    Tweets are tokenized using the tokenize function.

    This function is modified from spam.py from ps2.

    Args:
        tokens_list: A (n,1) numpy array of a list of tokens of every tweet.   

    Returns:
        words: A list of frequently occuring words. 
    """
    word_dict = {}
    words = []
    for tweet in tokens_list:
        for word in tweet:
            if word in string.punctuation:
                continue
            if word not in words:
                if word not in word_dict:
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
                    if word_dict[word] >= 5:
                        words.append(word)
    word_dict2 = {words[i]: i for i in range(len(words))}
    return word_dict2

def tokenize_words(df):
    """ Given a dataframe, tokenizes the tweet text using nltk tokenizer.
    Functionality included to stem words, but in experimentation, we found
    this to not improve accuracy.
    
    Returns:
        df: dataframe with new tokens column
    """
    # tokenize each sentence
    df['tokens'] = df.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)

    # stem each token 
    #s = PorterStemmer()
    #df['stemmed_tokens'] = df['tokens'].apply(lambda x: [s.stem(token) for token in x])
    return df

def remove_words(df, vocab):
    """ Removes words in dataframe that do not occur in dictionary.

    Dictionary created in creat_dictionary()

    Returns:
        df: dataframe with new 'in_dict' column of tokens that appear in the word list 
    """
    df['in_dict'] = df['tokens'].apply(lambda x: [word for word in x if word in vocab])
    return df

"""
Naive Bayes Functions
"""

def transform_text(tweets, vocab):
    """Transform a list of text messages into a pandas dataframe for further processing.

    This function creates a pandas dataframe that contains the number of times each word
    of the vocabulary appears in each message. 

    Each row in the resulting dataframe corresponds to the tweets.
    Each column corresponds to a word in the vocabulary. 

    This function is adapted from spam.py from ps2.

    Args:
        tweets: A numpy array of tweet tokens that appear in our vocabulary
        vocab: A python list of frequent words.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    t = np.zeros((len(tweets), len(vocab)))
    for i, tweet in enumerate(tweets):
        for word in tweet:
            if word in vocab:
                t[i][vocab[word]] += 1

    return t

def fit_naive_bayes_model_df(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    This function is adapted from spam.py from ps2.

    Args:
        transform: A pandas dataframe containing word counts for the training data and the
                    label for each tweet.

    Returns: The trained model as a Pandas Dataframe
    """
   
    n, V = matrix.shape

    matrix_y1_count = np.array([matrix[i][j] if labels[i]==1 else 0 for i in range(n) for j in range(V)]).reshape((n,V))
    matrix_y1 = np.sum(matrix_y1_count, axis=0)
    matrix_y0_count = np.array([matrix[i][j] if labels[i]==0 else 0 for i in range(n) for j in range(V)]).reshape((n,V))
    matrix_y0 = np.sum(matrix_y0_count, axis=0)

    phi_k_y1 = (matrix_y1 + 1) / (matrix_y1.sum() + V)
    phi_k_y0 = (matrix_y0 + 1) / (matrix_y0.sum() + V)

    return (phi_k_y1, phi_k_y0)

    
def predict_from_naive_bayes_model_df(model, prior, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function predicts on the models that fit_naive_bayes_model
    outputs.

    This function is adapted from spam.py from ps2.

    Args:
        model: A trained model from fit_naive_bayes_model
        prior: list of prior beliefs for each class
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model

                    p(x|y)p(y)
    p(y|x) = --------------------------     probabilty of type of tweet given the word
             p(x|y)p(y) + p(x|y)(1-p(y))
    """
    n, d = matrix.shape
    (phi_k_y1, phi_k_y0) = model

    preds = np.zeros((n,))
    l_y1 = np.sum(matrix * np.log(phi_k_y1), axis=1) + np.log(prior[1])
    l_y0 = np.sum(matrix * np.log(phi_k_y0), axis=1) + np.log(prior[0])
    preds = np.where(l_y1 > l_y0, 1, 0)
    return preds

def naive_bayes(train_df, test_df, vocab):
    """ Naive Bayes classifier adapted from from spam.py from ps2.

    Apply Bayes' theorem to classify the sentiment of tweets based on the
    word frequency.  

    Args:
        train_df: A pandas dataframe of the training dataset tokens and labels
        test_df: A pandas dataframe of the testing dataset tokens and labels
        vocab: A python list of frequent words.

    Returns:
        predictions: the classifier's predictions on the testing set in a list.
    """
    # create matrix of counts for both sets
    train_transform = transform_text(train_df['tokens'].to_numpy(), vocab)
    np.save("big_train_transform", train_transform)
    test_transform = transform_text(test_df['tokens'].to_numpy(), vocab)
    np.save("big_test_transform", test_transform)

    model = fit_naive_bayes_model_df(train_transform, train_df['label'].to_numpy())
    priors = train_df.groupby("label").size().div(len(train_df))

    predictions = predict_from_naive_bayes_model_df(model, priors, test_transform)
    
    naive_bayes_accuracy = np.mean(predictions == test_df['label'].to_numpy())
    print('Naive bayes analysis had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))
    return predictions

"""
n-gram functions
"""

def create_n_gram(tweet, n):
    """ Generate a list of n-grams from tweet text.

    Args:
        tweet: A list of normalized words from the message from tokenize_words().
        n: The number of grams to be generated.

    Returns:
       A list of n-grams from the tweet text.
    """
    temp=zip(*[tweet[i:] for i in range(0,n)])
    return np.array([" ".join(n_gram) for n_gram in temp])

def create_n_gram_model(tweets, labels, n=1):
    """
    Create a frequency distribution of n-grams across our set of tweets.

    Args:
        tweets: A list of strings containing tweet text
        labels: list of labels for tweets
        n: number of words to be in each n-gram

    Returns:
        A conditional frequency distribution for each n-gram and label
    """
    cfd = ConditionalFreqDist()
    # Look at all tweets, add up all n-gram frequency in table
    for i, tweet in enumerate(tweets):
        for gram in create_n_gram(tweet, n):
            cfd[gram][labels[i]] += 1

    # for each n-gram, divide the count per label by the total num occurances
    for gram in cfd:
        total = float(sum(cfd[gram].values()))
        for label in cfd[gram]:
            cfd[gram][label] /= total
    
    return cfd

def predict(model, tweets, word_dict, n):
    """ Use Conditional Frequency Distribution to compute predictions for a target matrix.

    This function predicts on the model that create_n_gram_model outputs.

    Args:
        model: CFD model from create_n_gram_model
        tweets: A list of strings containing tweet text
        n: number of words to be in each n-gram

    Returns:
        A conditional frequency distribution for each n-gram and label
    """
    final_predictions = []
    for tweet in tweets:
        predictions = [0, 0]
        words = [w for w in tweet if w in word_dict]
        for gram in create_n_gram(words, n):
            predictions = [predictions[i] + model[gram][i] for i in range(2)]
        final_predictions.append(np.argmax(predictions))
    return final_predictions

def n_gram(train_df, test_df, dictionary):
    """ n-gram classifier
    
    An n-gram model is a type of probabilistic language model that predicts 
    the next word in a sequence by considering the previous n words. 
    
    For example, a bigram model (n=2) would predict the next word by looking 
    at the two most recent words, while a trigram model (n=3) would use the 
    three most recent words. 

    Since n-grams capture the structure and patterns of language, we can classify
    the sentiment of a tweet based on the n-grams that are present.

    We found n = 2 to be the best number of grams.

    Args:
        train_df: A pandas dataframe of the training dataset tokens and labels
        test_df: A pandas dataframe of the testing dataset tokens and labels

    Returns:
        predictions: the classifier's predictions on the testing set in a list.
    """
    # number of grams
    n = 2
    print("n =", n)
    n_gram_model = create_n_gram_model(train_df['tokens'].to_list(), train_df['label'].to_list(), n)

    predictions = predict(n_gram_model, test_df['tokens'].to_list(), dictionary, n)
    n_gram_accuracy = [predictions[i] == test_df['label'].to_list()[i] for i in range(len(predictions))].count(True)/len(predictions)
    #n_gram_accuracy = np.mean(predictions == test_twt_labels) doesn't work
    print('n_gram analysis had an accuracy of {} on the testing set'.format(n_gram_accuracy))
    return predictions

"""
w2v functions
"""

def generate_model(tokens):
    """ Generate word2vec model based on the tokenized training set.
    
    Uses the Gensim word embedding: https://radimrehurek.com/gensim/models/word2vec.html

    Args:
        tokens: A numpy array of tweet tokens that appear in our vocabulary

    """
    w2v_m = Word2Vec(tokens, min_count = 1)
    #w2v_m.save('w2v_auto.model')
    return w2v_m

def w2v_nb(train_df, test_df, vocab):
    w2v_m = generate_model(train_df['tokens'].to_numpy())

    model_df = train_df['tokens'].apply(lambda x: np.mean([w2v_m.wv[token] for token in x], axis=0)).to_frame()
    model_df_norm = pd.DataFrame(model_df['tokens'].to_list())
    model_df_norm['label'] = train_df['label']
    clf = fit_naive_bayes_model_df(model_df_norm)

    priors = train_df.groupby("label").size().div(len(train_df))

    model_df_t = test_df['tokens'].apply(lambda x: np.mean([w2v_m.wv[token] for token in x if token in vocab], axis=0)).to_frame()
    model_df_norm_t = pd.DataFrame(model_df_t['tokens'].to_list())

    predictions = predict_from_naive_bayes_model_df(clf, priors, model_df_norm_t)
    w2v_accuracy = [predictions[i] == test_df['label'].to_list()[i] for i in range(len(predictions))].count(True)/len(predictions)
    print('word2vec analysis had an accuracy of {} on the testing set'.format(w2v_accuracy))
    return predictions

def train_w2v_model(train_df, train_labels, w2v_m):
    """ Train Decision Tree Classifier on word embedding vectors
    
    Args:
        train_df: A pandas dataframe of the training dataset tokens and labels
        test_df: A pandas dataframe of the testing dataset tokens and labels
        w2v_m: Word2Vec model generated by generate_model()

    Returns:
        clf: A decision tree classifier. 
    """
    clf = DecisionTreeClassifier()
    # each w2v_m.wv[token] is a n-dim vector encoding of the token word. This takes the mean of all words 
    # across all dimensions of this vector encoding
    model_df = train_df['tokens'].apply(lambda x: np.mean([w2v_m.wv[token] for token in x], axis=0)).to_frame()
    model_df_norm = pd.DataFrame(model_df['tokens'].to_list())
    clf.fit(model_df_norm, train_labels)
    print(clf.tree_.node_count)
    return clf

def predict_w2v_model(clf, w2v_m, test_df, vocab):
    """
    Predict sentiment on test set using Decision Tree and w2v embeddings
    
    Args:
        clf: A decision tree classifier from train_w2v_model()
        w2v_m: Word2Vec model generated by generate_model()
        test_df: A pandas dataframe of the testing dataset tokens
        vocab: list of frequent words        

    Returns:
        clf: A decision tree classifier. 
    """
    model_df = test_df['tokens'].apply(lambda x: np.mean([w2v_m.wv[token] for token in x if token in vocab], axis=0)).to_frame()
    model_df_norm = pd.DataFrame(model_df['tokens'].to_list())
    return clf.predict(model_df_norm)

def w2v(train_df, test_df, vocab):
    """ Decision tree classifier trained on word2vec features.

    Word2vec is a method for representing words as numerical vectors, 
    which can capture the meaning and context of words in a way that 
    allows them to be compared.

    Decision tree classification is a machine learning method for 
    classifying data into one of several possible categories. The 
    algorithm divides the data into smaller groups based on 
    specific rules, until each group only contains one type of data
    (i.e. it is homogenous)
    
    The rules for splitting the data are determined by the 
    algorithm, and are based on the characteristics of the data. Once the
    data has been split into homogeneous groups, the algorithm can make 
    predictions about new data based on the group it belongs to. 

    Args:
        train_df: A pandas dataframe of the training dataset tokens and labels
        test_df: A pandas dataframe of the testing dataset tokens and labels

    Returns:
        predictions: the classifier's predictions on the testing set in a list.
    """
    w2v_m = generate_model(train_df['tokens'].to_numpy())
    print("Generated w2v model.")
    #w2v_m = Word2Vec.load('w2v_auto.model')

    clf = train_w2v_model(train_df['tokens'].to_frame(), train_df['label'].to_list(), w2v_m)
    print("Fit model to training set.")

    predictions = predict_w2v_model(clf, w2v_m, test_df['tokens'].to_frame(), vocab)
    
    w2v_accuracy = np.mean(predictions == test_df['label'].to_list())
    #[predictions[i] == test_df['label'].to_list()[i] for i in range(len(predictions))].count(True)/len(predictions)
    print('word2vec analysis had an accuracy of {} on the testing set'.format(w2v_accuracy))
    return predictions

def main():
    train_df, test_df = get_tweets_and_labels()

    print("Number of train tweets: ", len(train_df['text']))
    print("Number of test tweets: ", len(test_df['text']))

    #Tokenization splits a message into tokens, normalizes them, and creates
    # a column in the dataframe.
    train_df['tokens'] = train_df.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)
    test_df['tokens'] = test_df.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)

    dictionary = create_dictionary(train_df['tokens'].to_numpy())
    print('Size of dictionary: ', len(dictionary))

    y_test = test_df['label'].to_list()
    """
    nb_pred = naive_bayes(train_df, test_df, dictionary)
    confusion_matrix = metrics.confusion_matrix(y_test, nb_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.show()

    ng_pred = n_gram(train_df, test_df, dictionary)
    confusion_matrix = metrics.confusion_matrix(y_test, ng_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.show()
    """
    w2_pred = w2v(train_df, test_df, dictionary)
    confusion_matrix = metrics.confusion_matrix(y_test, w2_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.show()
    return

if __name__ == '__main__':
    main()


    