import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def load_dataset(path):
    # Carichiamo l'intero dataset
    data = pd.read_csv(path)

    # otteniamo i titoli e le etichette
    headlines = data['headline'].values
    labels = data['clickbait'].values

    return data, headlines, labels


# Funzione per creare un grafico a barre che mostra
# la ripatizione in classi dei campioni
def show_datalet(data):
    sns.set_theme(style="darkgrid")
    ax = sns.countplot(x="clickbait", data=data)
    plt.savefig("../log/data_classes.png")


# Funzione per creare una word cloud
def create_word_cloud(data):
    headlines = data['headline'].values
    text = ""
    for word in headlines:
        text += word
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("../log/word_cloud.png")


if __name__ == '__main__':
    data, headlines, labels = load_dataset("../data/clickbait_data.csv")

    show_datalet(data)
    create_word_cloud(data)

    # dividiamo i dati di training e testing
    headlines_train, headlines_test, labels_train, labels_test = train_test_split(headlines, labels)
    print(f"Campioni di training: {headlines_train.shape}\nCampioni di testing: {headlines_test.shape}")
