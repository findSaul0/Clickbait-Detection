import threading

import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def show_datalet(data):
    sns.set_theme(style="darkgrid")
    ax = sns.countplot(x="clickbait", data=data)
    plt.savefig("../log/data_classes.png")


if __name__ == '__main__':
    # Carichiamo l'intero dataset
    DATASET_PATH = "../data/clickbait_data.csv"
    data = pd.read_csv(DATASET_PATH)

    # otteniamo i titoli e le etichette
    headlines = data['headline'].values
    labels = data['clickbait'].values

    # Mostriamo la ripartizione
    show_datalet(data)

    # dividiamo i dati di training e testing
    headlines_train, headlines_test, labels_train, labels_test = train_test_split(headlines, labels)
    print(f"Campioni di training: {headlines_train.shape}\nCampioni di testing: {headlines_train.shape}")
