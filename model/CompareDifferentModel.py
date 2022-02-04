import matplotlib.pyplot as plt
import pandas as pd

import utils.ProcessingData as ProcessingData

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
import warnings

warnings.filterwarnings('ignore')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from scipy import sparse


def tokenize(text):
    text = [word_tokenize(x) for x in text]
    return text


def run():
    # Carichiamo il dataset
    data = pd.read_csv('../data/dataset_in_wild_with_prediction.csv', index_col=0)

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

    # Creiamo dei grafici che mostrano la distribuzione dei campioni in base alle etichette
    ProcessingData.show_samples_distribution(data)
    plt.show()

"""    ### SEZIONE MODELLI ###
    data = pd.read_csv('../data/clickbait_data.csv', index_col=0)
    print(data.shape, "\n")
    # Create stopwords list
    stopwords_list = stopwords.words('english')
    # definining y and features
    features = data.drop(columns='clickbait')
    y = data['clickbait']
    # classes are mostly balanced
    print(y.value_counts(), "\n")
    # first splitting data for test/train sets
    # ngram range -> unigrams and bigrams
    X_train, X_test, y_train, y_test = train_test_split(features, y, random_state=20)
    tfidf = TfidfVectorizer(stop_words=stopwords_list, ngram_range=(1, 2))
    tfidf_text_train = tfidf.fit_transform(X_train['headline'])
    tfidf_text_test = tfidf.transform(X_test['headline'])
    X_train_ef = X_train.drop(columns='headline')
    X_test_ef = X_test.drop(columns='headline')
    # combine tf-idf vectors with the engineered features and store as sparse arrays
    X_train = sparse.hstack([X_train_ef, tfidf_text_train]).tocsr()
    X_test = sparse.hstack([X_test_ef, tfidf_text_test]).tocsr()
    print(X_train.shape)
    print(X_test.shape, "\n")

    ### DUMMY CLASSIFIER ###
    ProcessingData.dummy_classifier(X_train, y_train, X_test, y_test)
    ### NAIVE BAYES ###
    ProcessingData.naive_bayes(X_train, y_train, X_test, y_test)
    ### RANDOM FOREST ###
    ProcessingData.random_forest(X_train, y_train, X_test, y_test)
    ### SVM CLASSIFIER ###
    ProcessingData.svm_classifier(X_train, y_train, X_test, y_test)
    ### LOGISTIC REGRESSION ###
    ProcessingData.logistic_regression(X_train, y_train, X_test, y_test)
    ### XGBoost CLASSIFIER ###
    ProcessingData.XGBoost(X_train, y_train, X_test, y_test)

    plt.show()"""


if __name__ == '__main__':
    run()
