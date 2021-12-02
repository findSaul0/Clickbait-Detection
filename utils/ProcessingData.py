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
def show_data_classes(data):
    sns.set_style('darkgrid')
    plt.figure(figsize=(7, 5))
    fig1 = sns.countplot(data['clickbait'])
    plt.title('Clickbait vs Non-Clickbait')
    plt.ylabel('# of Headline')
    plt.xlabel('Type of Headline')
    fig1.set(xticklabels=['Non-Clickbait', 'Clickbait'])
    plt.tight_layout()
    #p lt.savefig("../log/data_classes.png")

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
    plt.savefig('../log/clickbait_20_bar')

def show_most_frequency_nonclickbait_word(words, counts):
    word_freq_figure1 = plt.figure(figsize=(10, 6))
    sns.barplot(words, counts, palette='Blues_d')
    plt.xticks(fontsize=16)
    plt.xticks(rotation=80)
    plt.title('Top 20 Non-Clickbait Headline Words')
    plt.xlabel('Most Common Words')
    plt.ylabel('Word Count')
    sns.set_style('darkgrid')
    plt.savefig('../log/nonclickbait_20_bar')


# Funzione per creare una word cloud
def create_word_clouds(data1, data2):
    wordcloud = WordCloud(colormap='Spectral').generate_from_frequencies(data1)
    plt.figure(figsize=(10, 10), facecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    #plt.savefig('../log/clickbait_wordcloud')

    wordcloud = WordCloud(colormap='Pastel2').generate_from_frequencies(data2)
    plt.figure(figsize=(10, 10), facecolor='k')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    # plt.savefig('../log/nonclickbait_wordcloud')
