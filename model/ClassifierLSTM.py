import os

from utils import ProcessingData

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model



from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Parameters
DATASET_PATH = "../data/clickbait_data.csv"
vocab_size = 10000
maxlen = 500
embedding_size = 32
batch_size = 512
epochs = 20
callbacks = [
    EarlyStopping(  # EarlyStopping is used to stop at the epoch where val_accuracy does not improve significantly
        monitor='val_accuracy',
        min_delta=1e-4,
        patience=4,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='saved/weights.h5',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
]


def training(model, X_train, Y_train, X_val, Y_val):
    history = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_val, Y_val), epochs=epochs,
                        callbacks=callbacks, shuffle=True)
    model.load_weights('saved/weights.h5')
    model.save('saved/model.hdf5')
    return history


def tokenization(X_train, X_val, X_test, headline):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(headline)

    with tf.device("/GPU:0"):
        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)
        X_val = tokenizer.texts_to_sequences(X_val)

        X_train = pad_sequences(X_train, maxlen=maxlen)
        X_test = pad_sequences(X_test, maxlen=maxlen)
        X_val = pad_sequences(X_val, maxlen=maxlen)

        return X_train, X_test, X_val


def create_model():
    with tf.device("/GPU:0"):
        # DEFINE AND TRAIN MODEL
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_size, input_length=maxlen))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(
            GlobalMaxPooling1D())  # Pooling Layer decreases sensitivity to features, thereby creating more generalised data for better test results.
        model.add(Dense(1024))
        model.add(Dropout(
            0.25))  # Dropout layer nullifies certain random input values to generate a more general dataset and prevent the problem of overfitting.
        model.add(Dense(512))
        model.add(Dropout(0.25))
        model.add(Dense(256))
        model.add(Dropout(0.25))
        model.add(Dense(128))
        model.add(Dropout(0.25))
        model.add(Dense(64))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

    plot_model(model, to_file="../log/model.png")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def show_training_result(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("../log/result.png")


def testing(model):
    model = load_model('saved/model.hdf5')
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    test_dataset = test_dataset.batch(64)
    result = model.evaluate(test_dataset)
    print("Accuracy: ", result[1])
    preds = [round(i[0]) for i in model.predict(X_test)]

    cm = ProcessingData.confusion_matrix_general(Y_test, preds, "../log/model_confusionmatrix.png")

    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("Recall of the model is {:.6f}".format(recall))
    print("Precision of the model is {:.6f}".format(precision))
    print("F1 score: {:.6f}".format(f1_score))



if __name__ == '__main__':
    # Carichiamo il dataset
    _, headline, labels = ProcessingData.load_dataset(DATASET_PATH)

    # Split data in train, validation and test set
    X_train, X_rem, Y_train, Y_rem = train_test_split(headline, labels, train_size=0.8)  # train 80%
    X_val, X_test, Y_val, Y_test = train_test_split(X_rem, Y_rem,
                                                    test_size=0.5)  # test and validation 10%

    # Print sets size
    print(f"Training - headline: {X_train.shape[0]}\tlabels: {Y_train.shape[0]}")
    print(f"Validation - headline: {X_val.shape[0]}\tlabels: {Y_val.shape[0]}")
    print(f"Testing - headline: {X_test.shape[0]}\tlabels: {Y_test.shape[0]}")

    # Tokenize headline
    X_train, X_test, X_val = tokenization(X_train, X_val, X_test, headline)

    # Create model
    model = create_model()

    # Train the model
    """history = training(model, X_train, Y_train, X_val, Y_val)
    show_training_result(history)"""

    # Testing the model
    testing(model)
