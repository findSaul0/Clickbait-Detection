import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.svm import LinearSVC

from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression
import time


def load_dataset(path):
    # Carichiamo l'intero dataset
    data = pd.read_csv(path)

    # otteniamo i titoli e le etichette
    headlines = data['headline'].values
    labels = data['clickbait'].values

    return data, headlines, labels


# Funzione per creare un grafico a barre che mostra
# la ripatizione in classi dei campioni
def show_data_classes(data):
    sns.set_style('darkgrid')
    plt.figure(figsize=(7, 5))
    fig1 = sns.countplot(data['clickbait'])
    plt.title('Clickbait vs Non-Clickbait')
    plt.ylabel('# of Headline')
    plt.xlabel('Type of Headline')
    fig1.set(xticklabels=['Non-Clickbait', 'Clickbait'])
    plt.tight_layout()
    # plt.savefig("../log/data_classes.png")


def show_most_frequency_clickbait_word(words, counts):
    plt.style.use('seaborn-talk')
    word_freq_figure1 = plt.figure(figsize=(10, 6))
    sns.barplot(words, counts, palette='Oranges_d')
    plt.xticks(fontsize=16)
    plt.xticks(rotation=80)
    plt.title('Top 20 Clickbait Headline Words')
    plt.xlabel('Most Common Words')
    plt.ylabel('Word Count')
    sns.set_style('darkgrid')
    # plt.savefig('../log/clickbait_20_bar')


def show_most_frequency_nonclickbait_word(words, counts):
    word_freq_figure1 = plt.figure(figsize=(10, 6))
    sns.barplot(words, counts, palette='Blues_d')
    plt.xticks(fontsize=16)
    plt.xticks(rotation=80)
    plt.title('Top 20 Non-Clickbait Headline Words')
    plt.xlabel('Most Common Words')
    plt.ylabel('Word Count')
    sns.set_style('darkgrid')
    # plt.savefig('../log/nonclickbait_20_bar')


# Funzione per creare una word cloud
def create_word_clouds(data1, data2):
    wordcloud = WordCloud(colormap='Spectral').generate_from_frequencies(data1)
    plt.figure(figsize=(10, 10), facecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    # plt.savefig('../log/clickbait_wordcloud')

    wordcloud = WordCloud(colormap='Pastel2').generate_from_frequencies(data2)
    plt.figure(figsize=(10, 10), facecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    # plt.savefig('../log/nonclickbait_wordcloud')


def show_samples_distribution(data):
    sns.set_style('darkgrid')

    # Creiamo un grafico per la distribuzione dei campioni in base alla feature "question"
    # Questo grafico mostrerà quanti titoli clickbait e non-clickbait sono domande o meno.
    plot = data.groupby('question')['clickbait'].value_counts().unstack().plot.bar(rot=0)
    plot.set_xlabel('Headline is a question?')
    plot.set_ylabel('Num of Headlines')
    plot.legend(title=None, labels=['Non-Clickbait', 'Clickbait'])
    plot.set(xticklabels=['No', 'Yes'])
    # plt.savefig('../log/question_distribution.png')

    # Creiamo un grafico per la distribuzione dei campioni in base alla feature "starts_with_num"
    # Questo grafico mostrerà quantit titoli clickbait e non-clickbait iniziano con un numero o meno
    plot = data.groupby('starts_with_num')['clickbait'].value_counts().unstack().plot.bar(rot=0)
    plot.set_xlabel('Headline starts with a number?')
    plot.set_ylabel('Num of Headlines')
    plot.legend(title=None, labels=['Non-Clickbait', 'Clickbait'])
    plot.set(xticklabels=['No', 'Yes'])
    # plt.savefig('../log/starts_with_number_distribution.png')

    # Creiamo un grafico per la distribuzione dei campioni in base alla feature "exclamation"
    # Questo grafico mostrerà quantit titoli clickbait e non-clickbait iniziano con un numero o meno
    plot = data.groupby('exclamation')['clickbait'].value_counts().unstack().plot.bar(rot=0)
    plot.set_xlabel('Headline contains exclamation mark?')
    plot.set_ylabel('Num of Headlines')
    plot.legend(title=None, labels=['Non-Clickbait', 'Clickbait'])
    plot.set(xticklabels=['No', 'Yes'])
    # plt.savefig('../log/exclamation_distribution.png')

    # Creiamo un grafico per la distribuzione dei campioni in base alla feature "headline_words"
    # Questo grafico mostrerà la distribuzione titoli clickbait e non-clickbait in base al numero di parole
    plot = data.groupby('headline_words')['clickbait'].value_counts().unstack().plot.bar(rot=0)
    plot.set_xlabel('Num of Words')
    plot.set_ylabel('Num of Headlines')
    plot.legend(title=None, labels=['Non-Clickbait', 'Clickbait'], loc='upper right')
    # plt.savefig('../log/headline_words_distribution.png')


# creating a function to call after each model iteration to print accuracy and recall scores for test and train
def train_results(preds, y_train):
    return "Training Accuracy:", accuracy_score(y_train, preds), "Training Recall:", recall_score(y_train, preds)


def test_results(preds, y_test):
    return "Testing Accuracy:", accuracy_score(y_test, preds), "Testing Recall:", recall_score(y_test, preds), \
           "Testing Precision:", precision_score(y_test, preds), "Testing F1 Score:", f1_score(y_test, preds)


def dummy_classifier(X_train, y_train, X_test, y_test):
    # baseline model to predict majority class
    dc_classifier = DummyClassifier(strategy='most_frequent')
    dc_classifier.fit(X_train, y_train)
    dc_train_preds = dc_classifier.predict(X_train)
    dc_test_preds = dc_classifier.predict(X_test)
    print("DUMMY CLASSIFIER")
    print(train_results(dc_train_preds, y_train))
    print(test_results(dc_test_preds, y_test))

    confusion_matrix_general(y_test, dc_test_preds, "../log/dummyclassifier_confusionmatrix")


def naive_bayes(X_train, y_train, X_test, y_test):
    nb_classifier = MultinomialNB(alpha=.05)
    nb_classifier.fit(X_train, y_train)
    nb_train_preds = nb_classifier.predict(X_train)
    nb_test_preds = nb_classifier.predict(X_test)
    print("NAIVE BAYES CLASSIFIER")
    print(train_results(nb_train_preds, y_train))
    print(test_results(nb_test_preds, y_test))
    confusion_matrix_general(y_test, nb_test_preds, "../log/naivebayes_confusionmatrix.png")


def random_forest(X_train, y_train, X_test, y_test):
    rf_classifier = RandomForestClassifier(class_weight='balanced', n_estimators=900)
    rf_classifier.fit(X_train, y_train)
    rf_test_preds = rf_classifier.predict(X_test)
    rf_train_preds = rf_classifier.predict(X_train)
    print("RANDOM FOREST")
    print(train_results(rf_train_preds, y_train))
    print(test_results(rf_test_preds, y_test))
    confusion_matrix_general(y_test, rf_test_preds, "../log/randomforest_confusionmatrix.png")


def svm_classifier(X_train, y_train, X_test, y_test):
    svm_classifier = LinearSVC(class_weight='balanced', C=10, max_iter=1500)
    svm_classifier.fit(X_train, y_train)
    svm_test_preds = svm_classifier.predict(X_test)
    svm_train_preds = svm_classifier.predict(X_train)
    print("SVM CLASSIFIER")
    print(train_results(svm_train_preds, y_train))
    print(test_results(svm_test_preds, y_test))
    confusion_matrix_general(y_test, svm_test_preds, "../log/svm_confusionmatrix.png")


def logistic_regression(X_train, y_train, X_test, y_test):
    lr = LogisticRegression(C=500, class_weight='balanced', solver='liblinear', tol=0.0001)
    lr.fit(X_train, y_train)
    lr_train_preds = lr.predict(X_train)
    lr_test_preds = lr.predict(X_test)
    print("LOGISTIC REGRESSION")
    print(train_results(lr_train_preds, y_train))
    print(test_results(lr_test_preds, y_test))
    confusion_matrix_general(y_test, lr_test_preds, "../log/logisticregression_confusionmatrix.png")


def XGBoost(X_train, y_train, X_test, y_test):
    xgb_clf = XGBClassifier()
    xgb_clf.fit(X_train, y_train)
    xgb_test_preds = xgb_clf.predict(X_test)
    xgb_train_preds = xgb_clf.predict(X_train)
    print("XGBOOST CLASSIFIER")
    print(test_results(xgb_test_preds, y_test))
    print(train_results(xgb_train_preds, y_train))
    confusion_matrix_general(y_test, xgb_test_preds, "../log/xgboost_confusionmatrix.png")


def confusion_matrix_general(y_test, preds, path):
    # plot confusion matrix on test set Dummy Classifier
    plt.figure()
    sns.set()
    cm_dc = confusion_matrix(y_test, preds)
    sns.heatmap(cm_dc.T, square=True, annot=True, fmt='d', cbar=False, cmap="inferno",
                xticklabels=['non-clickbait', 'clickbait'], yticklabels=['non-clickbait', 'clickbait'])
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.tight_layout()
    plt.savefig(path)

    return cm_dc
