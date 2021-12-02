import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import string as s
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score,accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os


### VIEW MODEL ###
FILE_PATH = "../data/clickbait_data.csv"
data = pd.read_csv(FILE_PATH)
print(data, "\n")

### SPLITTING DATASET INTO TRAIN AND TEST SETS ###
x=data.headline
y=data.clickbait
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.25,random_state=22,stratify=data['clickbait'])

### ANALYZING TRAIN AND TEST DATA  ###
print("No. of elements in training set")
print(train_x.size, "\n")
print("No. of elements in testing set")
print(test_x.size, "\n")

print(train_x.head(), "\n")
print(train_y.head(), "\n")

print(test_x.head(), "\n")
print(test_y.head(), "\n")

from utils.ClearDataset import tokenization, lowercasing, remove_stopwords, remove_punctuations, remove_numbers,remove_spaces, lemmatzation


train_x=train_x.apply(tokenization)
test_x=test_x.apply(tokenization)

train_x=train_x.apply(lowercasing)
test_x=test_x.apply(lowercasing)

train_x=train_x.apply(remove_stopwords)
test_x=test_x.apply(remove_stopwords)

train_x=train_x.apply(remove_punctuations)
test_x=test_x.apply(remove_punctuations)

train_x=train_x.apply(remove_numbers)
test_x=test_x.apply(remove_numbers)

train_x=train_x.apply(remove_spaces)
test_x=test_x.apply(remove_spaces)

print(train_x.head(), "\n")
print(test_x.head(), "\n")

train_x=train_x.apply(lemmatzation)
test_x=test_x.apply(lemmatzation)


train_x=train_x.apply(lambda x: ''.join(i+' ' for i in x))
test_x=test_x.apply(lambda x: ''.join(i+' ' for i in x))


### COUNTVECTORISER ###
cov=TfidfVectorizer(analyzer='word', ngram_range=(1,2),max_features=22500)
train_1=cov.fit_transform(train_x)
test_1=cov.transform(test_x)

train_arr=train_1.toarray()
test_arr=test_1.toarray()


### DEFINE NAIVE BAYES CLASSIFIER AND TRAINING ###
NB_MN=MultinomialNB()
NB_MN.fit(train_arr,train_y)
pred=NB_MN.predict(test_arr)

### EVALUATION  OF RESULT ###
print("F1 score of the model")
print(f1_score(test_y,pred))
print("Accuracy of the model")
print(accuracy_score(test_y,pred))
print("Accuracy of the model in percentage")
print(accuracy_score(test_y,pred)*100,"%", "\n")


print("Confusion Matrix")
print(confusion_matrix(test_y,pred))

print("Classification Report")
print(classification_report(test_y,pred), "\n")

sns.set(font_scale=1.5)
cof=confusion_matrix(test_y, pred)
cof=pd.DataFrame(cof, index=[i for i in range(2)], columns=[i for i in range(2)])
plt.figure(figsize=(8,8))

sns.heatmap(cof, cmap="PuRd",linewidths=1, annot=True,square=True,cbar=False,fmt='d',xticklabels=['Non-clickbait','Clickbait'],yticklabels=['Non-clickbait','Clickbait'])
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.title("Confusion Matrix for Clickbait Classification")
plt.show()

