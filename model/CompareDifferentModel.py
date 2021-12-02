import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
import numpy as np
import warnings

warnings.filterwarnings('ignore')

import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score

from sklearn import svm
from sklearn.svm import LinearSVC

from sklearn.dummy import DummyClassifier

from wordcloud import WordCloud, STOPWORDS

import string
from matplotlib import style


def tokenize(text):
    text = [word_tokenize(x) for x in text]
    return text


def run():
    # Carichiamo il dataset
    data = pd.read_csv('../data/clickbait_data.csv', index_col=0)

    # Visualizziamo la frequenza delle varie classi nel dataset
    sns.set_style('darkgrid')
    plt.figure(figsize=(7, 5))
    fig1 = sns.countplot(data['clickbait'])
    plt.title('Clickbait vs Non-Clickbait')
    plt.ylabel('# of Headline')
    plt.xlabel('Type of Headline')
    fig1.set(xticklabels=['Non-Clickbait', 'Clickbait'])
    plt.tight_layout()
    plt.savefig("../log/data_classes.png")

    # Puliamo i dati rimuovendo le stopword e dividendo il testo in parole
    data.headline = tokenize(data.headline)

    stopword_list = stopwords.words('english')
    data.headline = data['headline'].apply(lambda x: [item for item in x if item not in stopword_list])

    # Creiamo due dataframe separate per gli articoli clickbait e non clickbai
    data_clickbait = data[data['clickbait'] == 1]
    data_non_clickbait = data[data['clickbait'] == 0]

    # Creiamo una lista di parole uniche per ogni classe così da individuare le più frequenti
    cb_word_list = list(data_clickbait['headline'])
    vocab_cb = set()
    for word in cb_word_list:
        vocab_cb.update(word)
    print(len(vocab_cb))

    noncb_word_list = list(data_non_clickbait['headline'])
    vocab_noncb = set()
    for word in noncb_word_list:
        vocab_noncb.update(word)
    print(len(vocab_noncb))

    flat_cb = [item for sublist in cb_word_list for item in sublist]
    flat_noncb = [item for sublist in noncb_word_list for item in sublist]

    cb_freq = FreqDist(flat_cb)
    noncb_freq = FreqDist(flat_noncb)

    # create counts of clickbait and non-clickbait words and values
    cb_bar_counts = [x[1] for x in cb_freq.most_common(20)]
    cb_bar_words = [x[0] for x in cb_freq.most_common(20)]

    noncb_bar_counts = [x[1] for x in noncb_freq.most_common(20)]
    noncb_bar_words = [x[0] for x in noncb_freq.most_common(20)]

    plt.style.use('seaborn-talk')

    # bar plot for top 15 most common clickbait words
    word_freq_figure1 = plt.figure(figsize=(10, 6))
    sns.barplot(cb_bar_words, cb_bar_counts, palette='Oranges_d')
    plt.xticks(fontsize=16)
    plt.xticks(rotation=80)
    plt.title('Top 20 Clickbait Headline Words')
    plt.xlabel('Most Common Words')
    plt.ylabel('Word Count')
    sns.set_style('white')
    plt.savefig('../clickbait_20_bar')
    plt.show()

if __name__ == '__main__':
    run()
