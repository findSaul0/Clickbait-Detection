import numpy as np
import pandas as pd

from ClassifierLSTM import create_model

from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Carichiamo il modello
    model = load_model('saved/model.hdf5')

    # Carichiamo il dataset per il testing in the wild
    data = pd.read_csv("../data/dataset_in_wild.csv")
    headline = data['headline'].values

    # Settiamo il tokenizer
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(headline)

    # Tokenizziamo i titoli di giornale
    headline = tokenizer.texts_to_sequences(headline)
    headline = pad_sequences(headline, maxlen=500)

    # Classifichiamo i titoli di giornale
    preds = [round(i[0]) for i in model.predict(headline)]

    # Inseriamo le predizioni come colonna nel database
    data.insert(2, "clickbait", preds)
    data.to_csv("../data/dataset_in_wild_with_prediction.csv", index=False)

    # Costruiamo un dizionario dove le coppie chiave-valore sono giornali-numero_articoli_clickbait
    newspaper = {}
    for paper in data["newspaper"].values:
        newspaper[paper] = 0

    for index, row in data.iterrows():
        newspaper[row["newspaper"]] = newspaper[row["newspaper"]] + row["clickbait"]

    for key, value in newspaper.items():
        print(f"{key}: {value}")