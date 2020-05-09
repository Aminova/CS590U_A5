import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

cough_data = "coughData.json"

def load_sound_files(dpath):
    with open(dpath, "r") as fp:
        data = json.load(fp)
    mfcc = np.array(data["mfcc"])
    labels = np.array(data["labels"])
    return mfcc, labels


def plot_history(modelfit):
    fig, axes = plt.subplots(2)

    axes[0].plot(modelfit.history["accuracy"], label="training accuracy")
    axes[0].plot(modelfit.history["val_accuracy"], label="testing accuracy")
    axes[0].set_ylabel("Accuracy")
   
    axes[1].plot(modelfit.history["loss"], label="training loss")
    axes[1].plot(modelfit.history["val_loss"], label="testing loss")
    axes[1].set_ylabel("Error")
   
    axes[1].set_xlabel("Epoch")
   
    plt.show()


def splitData(test_size, testing_size):
    X, y = load_sound_files(cough_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=testing_size)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_cnn_model(input_shape):
   
    cough_model = keras.Sequential()
    cough_model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    cough_model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    cough_model.add(keras.layers.BatchNormalization())
   
    cough_model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    cough_model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    cough_model.add(keras.layers.BatchNormalization())
   
    cough_model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    cough_model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    cough_model.add(keras.layers.BatchNormalization())
   
    cough_model.add(keras.layers.Flatten())
    cough_model.add(keras.layers.Dense(64, activation='relu'))
   
    cough_model.add(keras.layers.Dropout(0.3))
    cough_model.add(keras.layers.Dense(2, activation='softmax'))

    return model

X_train, X_val, X_test, y_train, y_val, y_test = splitData(0.25, 0.2)
input_shape = (X_train.shape[1], X_train.shape[2], 1)

model = build_cnn_model(input_shape)

optimizer = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer,
              loss='BinaryCrossentropy',
              metrics=['accuracy'])

model.summary()

modelfit = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=30, epochs=25)

plot_history(modelfit)

testing_loss, testing_accuracy = model.evaluate(X_test, y_test, verbose=2)

print('\nTest accuracy:', testing_accuracy)