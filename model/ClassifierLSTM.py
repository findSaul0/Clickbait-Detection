import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPooling1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model

import utils.ProcessingData as ProcessingData

DATASET_PATH = "../data/clickbait_data.csv"
data, headline, labels = ProcessingData.load_dataset(DATASET_PATH)

# Split data in train, validation and test set
X_train, X_rem, Y_train, Y_rem = train_test_split(headline, labels, shuffle=True, train_size=0.8)  # train 80%
X_val, X_test, Y_val, Y_test = train_test_split(X_rem, Y_rem, shuffle=True, test_size=0.5)  # test and validation 10%

print(f"Training: {X_train.shape} - {Y_train.shape}")
print(f"Validation: {X_val.shape} - {Y_val.shape}")
print(f"Testing: {X_test.shape} - {Y_test.shape}")



vocab_size = 10000
maxlen = 500
embedding_size = 32

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(headline)

with tf.device("/GPU:0"):
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    X_val = tokenizer.texts_to_sequences(X_val)

    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)
    X_val = pad_sequences(X_val, maxlen=maxlen)


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

"""callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        min_delta=1e-4,
        patience=3,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='weights.h5',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
]"""

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(X_train.shape)
print(Y_train.shape)
history = model.fit(X_train, Y_train, batch_size=512, validation_data=(X_val, Y_val), epochs=20)

model.save('model')

# PLOT TRAINING METRICS
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

# PLOT CON
preds = [round(i[0]) for i in model.predict(X_test)]
cm = confusion_matrix(Y_test, preds)
plt.figure()
plot_confusion_matrix(cm, figsize=(12, 8), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(2), ['Not clickbait', 'Clickbait'], fontsize=16)
plt.yticks(range(2), ['Not clickbait', 'Clickbait'], fontsize=16)
plt.savefig("../log/matrix.png")

tn, fp, fn, tp = cm.ravel()

precision = tp / (tp + fp)
recall = tp / (tp + fn)

print("Recall of the model is {:.2f}".format(recall))
print("Precision of the model is {:.2f}".format(precision))
