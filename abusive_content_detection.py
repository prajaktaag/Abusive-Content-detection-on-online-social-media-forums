"""
    file        :   abusive_content_detection.py
    language    :   python3
    version     :   02/04/17/04/2017 9:38 AM, v1.0
    author      :   Vaibhav Nagda(vjn4006@rit.edu), Prajakta Gaydhani(pag3862@rit.edu), Virtee Parekh(vvp2639@rit.edu)
    description :   This file cleans the given data, and performs various machine learning algorithms
                    to detect abusive content in the given data.
"""

__author__ = ["vjn4006(Vaibhav Nagda)", "pag3862(Prajakta Gaydhani)", "vvp2639(Virtee Parekh)"]

#references :
#[1] http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
#[2] http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

import os
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import csv

DIR = 'data'
files = map(lambda x: os.path.join(DIR, x), os.listdir(DIR))
files = filter(lambda x: 'csv' in x, files)


def readCSV(file_name):
    """
    This function reads the CSV file.
    :param file_name: name of the file
    :return: comments : the comment column of the CSV file
    """
    data = pd.read_csv(file_name)
    data.columns = ['Insult', 'Date', 'Comments']
    comments = data['Comments']
    return comments

def clean_data(data):
    """
    This function cleans both training and testing data from the CSV files
    :param data: data (comments) to be cleaned
    :return: data:  the cleaned data
    """
    for i in range(0, len(data)):
        data[i] = data[i][1:-1]  # Removing first and last double inverted commas
        data[i] = data[i].lower()

        words_seperated_by_space = data[i].split(" ")
        words_seperated_by_space = [k.replace("\\xa0", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\\xc2", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\\n", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\\r", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\\xc8", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\\x9b", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\\x99", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\\xc4", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\\x83", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\"99%er\"", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\\x99", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("2x80", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("cxf3", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("2x80", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("u0111", " ") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("cixe2x80m", " ") for k in words_seperated_by_space]

        words_seperated_by_space = [k.replace("don't", "dont") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("won't", "wont") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("can't", "cant") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("i\'m", "i am") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("ain't", "is not") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\'ll", "will") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\'t", "not") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\'ve", "have") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("'s", "is") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\'re", "are") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\'d", "would") for k in words_seperated_by_space]
        words_seperated_by_space = [k.replace("\\ \\ ", " ") for k in words_seperated_by_space]

        words_seperated_by_space = [re.sub(" u ", "you", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub("[!]+", "!", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('[?]+', "?", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('[.]+', ".", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('[\\\]+', "", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('[\']+', ".", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub("(haha)+", "haha", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub("(l(ol)+)", "lol ", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub("(bw(a)+h)", "bwah", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub("(bwa(h)+)", "bwah", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub("(bwa(h)+)", "bwah", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub("<[a-z]*>", "", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub("[*****]+", "", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub("(xe2x80x(9|a6))", "", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('_', "", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('-', "", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('<>', "", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('g(rrr)+', "grr", str(k)) for k in words_seperated_by_space]
        words_seperated_by_space = [re.sub('u(mmm)+|u(mm)+', "umm", str(k)) for k in words_seperated_by_space]

        data[i] = ' '.join(words_seperated_by_space)

    return data


def get_features(file_name):
    """
    This function writes the cleaned data to the file.
    :param fileName: name of the CSV file to be read
    :return: None
    """
    data = pd.read_csv(file_name)
    data.columns = ['Insult', 'Date', 'Comment']
    data_insults = data['Insult'].tolist()
    data_comments = data['Comment'].tolist()
    data_dates = data['Date'].tolist()
    #function call to clean_data
    clean_data_comments = clean_data(data_comments)

    #write the cleaned data to CSV file
    file_name = file_name.strip('.csv')
    with open(os.path.join('cleaned_data', file_name + '_cleaned.csv'), 'w') as csvfile:
        fieldnames = ['Insult', 'Date', 'Comments']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(0, len(clean_data_comments)):
            writer.writerow({'Insult': data_insults[i], 'Date': data_dates[i], 'Comments': clean_data_comments[i]})


def read_CSV_Header_removal(fileName):
    """
    This function reads both training and testing data from the CSV files without the lables
    :param fileName: name of the CSV file to be read
    :return: data: data read from the CSV file as a list
    """
    with open(fileName, 'r') as file:
        Data = csv.reader(file)
        #skip the labels while reading the CSV file
        next(Data, None)
        Data = list(Data)
    return Data



def BagofWords(trainData, testData):
    """
    This function creates bag of words and feature extraction from trainig  and testing data for
    Multinomial Naive Bayes  model
    :param trainData: the training data
    :param testData: the testing data
    :return: trainFeatureData : feature vector for training data
    :return: trainFeatureData : feature vector for testing data
    :return: trainData : training data
    :return: testData : testing data
    """

    #CountVectorizer method remove all the stopwords, convert all the string into smaller case,
    # use unigram approach, count the occurences each word in the training data and generate Bag of Words
    vocabVector = CountVectorizer(analyzer='word', binary=True, lowercase=True, stop_words='english', vocabulary=None)

    #generate document- matrix for both trainings and testing data
    trainFeatureData = vocabVector.fit_transform([train_Data[2] for train_Data in trainData])
    testFeatureData = vocabVector.transform([test_Data[2] for test_Data in testData])

    return trainFeatureData, testFeatureData, trainData, testData


def NB_multinominal(trainFeatureData, testFeatureData, trainData, testData):
    """
    This function make predictions on the testing data to predict if the content is abusive or not
    Uses Multinomial Naive Bayes model for prediction.
    :param trainData: the training data
    :param testData: the testing data
    :param: trainFeatureData : feature vector for training data
    :param: trainFeatureData : feature vector for testing data
    :return: abusivetextPrediction : list of predictions made on the testing data
    """

    #creates object of Multinomail Naive Bayes.
    nb = MultinomialNB()

    # train the model using bagOfWords and classification for training set
    nb.fit(trainFeatureData, [(train_Data[0]) for train_Data in trainData])

    # make predictions for test data features
    abusivetextPrediction = nb.predict(testFeatureData)

    return list(abusivetextPrediction)

def modelAccurarcy(trainData, testData, abusivetextPrediction):
    """
    This function calculates the accuracy for Multinomial Naive Bayes and SVM model
    :param trainData: the training data
    :param testData: the testing data
    :param: abusivetextPrediction : list of predictions made on the testing data (estimated values)
    :return: modelAccuracy :  Accuracy of the Naive Bayes model
    """

    #calculates the overall accuracy of the model by comparing actual and predicted data
    modelAccuracy = accuracy_score([(test_data[0]) for test_data in testData], abusivetextPrediction) * 100
    return modelAccuracy

def confusionMatrix(testData, abusivetextPrediction):
    """
    This function displays the confusion matrix for Naive Bayes and SVM model
    :param testData: the testing data
    :param: abusivetextPrediction : list of predictions made on the testing data (estimated values)
    :return: None
    """
    print(confusion_matrix([(test_data[0]) for test_data in testData], abusivetextPrediction))

def calc_F1score(testData, abusivetextPrediction):
    """
    This function calculates precision, recall, f1_score and support for Naive Bayes and SVM model
    :param testData: the testing data
    :param: abusivetextPrediction : list of predictions made on the testing data (estimated values)
    :return: None
    """
    target_names = ['Not Abusive (0) ', 'Abusive (1)']
    print("\n")
    print(classification_report([(test_data[0]) for test_data in testData], abusivetextPrediction, target_names=target_names))

def SVM(trainData, testData):
    """
    This function make predictions on the testing data to predict if the content is abusive or not
    Uses SVM model for prediction.
    :param trainData: the training data
    :param testData: the testing data
    :return: prediction : list of predictions made on the testing data set
    """

    #Vectorize the data
    vectorizer = TfidfVectorizer(analyzer='word', binary=True, lowercase=True, stop_words='english', vocabulary=None,
                                 min_df=1)

    #extract features from training and testing data
    train_featureVecotor = vectorizer.fit_transform([train_Data[2] for train_Data in trainData])
    test_featureVector = vectorizer.transform([test_Data[2] for test_Data in testData])

    #creates an object for SVM model
    svmobj = SVC(C= 0.7, kernel='linear')
    #svmobj = SVC(C=10000.0, kernel='rbf')
    #svmobj = SVC(C=0.03, gamma = 0.00003, kernel='rbf')


    # train the model using bagOfWords and classification for training set
    svmobj.fit(train_featureVecotor, [train_Data[0] for train_Data in trainData])

    # make predictions for test data features
    prediction = svmobj.predict(test_featureVector)

    #return  a list of predictions made on testing set
    return list(prediction)

if __name__ == '__main__':
    """
    Main program
    """

    for f in files:
        get_features(f)
    DIR = 'cleaned_data/data'
    files = map(lambda x: os.path.join(DIR, x), os.listdir(DIR))
    train_data = []
    test_data = []
    vocab = []
    #cleaned data
    for f in files:
        if 'train' in f:
            train_data = read_CSV_Header_removal(f)
        elif 'solution' in f:
            test_data = read_CSV_Header_removal(f)

    #------Naive Bayes Model -----------------#

    #create bag of words for Naive Bayes model
    trainFeatureData, testFeatureData, trainData, testData = BagofWords(train_data, test_data)
    # function call to Naive Bayes model to make predictions on testing data set
    abusivetextPrediction = NB_multinominal(trainFeatureData, testFeatureData, train_data, test_data)
    #calculates the accuracy for Naive Bayes model
    modelAccurarcyNB = modelAccurarcy(train_data, test_data, abusivetextPrediction)
    print("MULTINOMIAL NAIVE BAYES")
    print("Model Accuracy Naive Bayes : ", modelAccurarcyNB)
    print("Confusion Matrix Naive Bayes  \n")
    #displays confusion matrix for Naive Bayes model
    confusionMatrix(test_data, abusivetextPrediction)
    #calculates score for Naive Bayes model
    calc_F1score(testData, abusivetextPrediction)

    print("\n--------------------------------------------------------------\n\n")

    # ------SVM Model -----------------#

    # function call to SVM model to make predictions on testing data set
    precdictionSVM = SVM(train_data, test_data)
    # calculates the accuracy for SVM
    modelAccuracySVM = modelAccurarcy(train_data, test_data, precdictionSVM)
    print("SUPPORT VECTOR MACHINE")
    print("Model Accuracy SVM : ", modelAccuracySVM)
    print("Confusion Matrix SVM \n")
    # displays confusion matrix for SVM model
    print(confusion_matrix([test_Data[0] for test_Data in testData], precdictionSVM))
    #calculates score for SVM model
    calc_F1score([test_Data[0] for test_Data in testData], precdictionSVM)






