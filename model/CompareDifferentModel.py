import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils.ProcessingData as ProcessingData

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
    ProcessingData.show_data_classes(data)

    # Puliamo i dati rimuovendo le stopword e dividendo il testo in parole
    data.headline = [word_tokenize(x) for x in data.headline]
    stopword_list = stopwords.words('english')
    data.headline = data['headline'].apply(lambda x: [item for item in x if item not in stopword_list])

    # Creiamo due dataframe separati per gli articoli clickbait e non clickbait
    df_clickbait = data[data['clickbait'] == 1]
    df_non_clickbait = data[data['clickbait'] == 0]

    # Creiamo una dizionario delle parole nei titoli clickbait
    cb_word_list = list(df_clickbait['headline'])
    vocab_cb = set()
    for word in cb_word_list:
        vocab_cb.update(word)
    print("Dimensione vocabolario Clickbait: ", len(vocab_cb))

    # Creiamo un dizionario delle parole nei titoli non clickbait
    noncb_word_list = list(df_non_clickbait['headline'])
    vocab_noncb = set()
    for word in noncb_word_list:
        vocab_noncb.update(word)
    print("Dimensione vocabolario Non-Clickbait: ", len(vocab_noncb))

    # Calcoliamo la frequenza delle parole
    flat_cb = [item for sublist in cb_word_list for item in sublist]
    flat_noncb = [item for sublist in noncb_word_list for item in sublist]
    cb_freq = FreqDist(flat_cb)
    noncb_freq = FreqDist(flat_noncb)

    # Individuiamo le 20 parole più frequenti sia nei titoli clickbait che non
    cb_bar_counts = [x[1] for x in cb_freq.most_common(20)]
    cb_bar_words = [x[0] for x in cb_freq.most_common(20)]

    noncb_bar_counts = [x[1] for x in noncb_freq.most_common(20)]
    noncb_bar_words = [x[0] for x in noncb_freq.most_common(20)]

    # Creiamo dei grafici a barre
    ProcessingData.show_most_frequency_clickbait_word(cb_bar_words, cb_bar_counts)
    ProcessingData.show_most_frequency_nonclickbait_word(noncb_bar_words, noncb_bar_counts)

    # Creiamo anche una nuvola di parole
    clickbait_dictionary = dict(zip(cb_bar_words, cb_bar_counts))
    nonclickbait_dictionary = dict(zip(noncb_bar_words, noncb_bar_counts))
    ProcessingData.create_word_clouds(clickbait_dictionary, nonclickbait_dictionary)

    plt.show()


if __name__ == '__main__':
    run()
