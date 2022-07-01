# -*- encoding:utf-8 -*-

import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

from utils import get_features


def create_model():
    model = keras.Sequential([
        keras.layers.Dense(100, activation='elu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(200, activation='elu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(100, activation='elu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax'),
    ])

    return model


def train_model(model, X_train, X_test, y_train, y_test):
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy'])

    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath='saved_models/audio_classification.hdf5',
        verbose=1, save_best_only=True)

    model.fit(X_train, y_train, batch_size=128, epochs=100,
              validation_data=(X_test, y_test),
              callbacks=[checkpointer], verbose=1)


def evaluate_model(model, X_test, y_test):
    test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(test_accuracy[1])


def predict(model, audio_file):
    feature = get_features(audio_file).reshape(1, -1)
    print(feature.shape)
    label_prob = model.predict(feature)
    print(label_prob)
    class_id = np.argmax(label_prob, axis=1)
    print(class_id)


def load_dataset():
    dataset = np.load("/data/workflow/data/UrbanSound8K/samples/dnn/samples.npy", allow_pickle=True)
    X = np.array([t.tolist() for t in dataset[:, 0]], dtype=np.float)
    y = np.array((dataset[:, 1]).tolist(), dtype=np.int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test


def main():
    model = create_model()
    X_train, X_test, y_train, y_test = load_dataset()
    train_model(model, X_train, X_test, y_train, y_test)
    evaluate_model(model, X_test, y_test)
    predict(model, "/data/workflow/data/UrbanSound8K/audio/fold8/103076-3-0-0.wav")


def sever_model(audio_file):
    model = keras.models.load_model('saved_models/audio_classification.hdf5')
    predict(model, audio_file)


def debug():
    model = keras.Sequential([
        keras.layers.Dense(100, activation='elu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(200, activation='elu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(100, activation='elu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax'),
    ])

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy'])

    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath='saved_models/audio_classification.hdf5',
        verbose=1, save_best_only=True)

    dataset = np.load("/data/workflow/data/UrbanSound8K/samples/dnn/samples.npy", allow_pickle=True)
    X = np.array([t.tolist() for t in dataset[:, 0]], dtype=np.float)
    y = np.array((dataset[:, 1]).tolist(), dtype=np.int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model.fit(X_train, y_train, batch_size=128, epochs=100,
              validation_data=(X_test, y_test),
              callbacks=[checkpointer], verbose=1)

    test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(test_accuracy[1])

    predict_x = model.predict(X_test)
    class_x = np.argmax(predict_x, axis=1)

    audio_file = "/data/workflow/data/UrbanSound8K/audio/fold8/103076-3-0-0.wav"
    feature = get_features(audio_file).reshape(1, -1)

    print(feature.shape)
    label_prob = model.predict(feature)
    print(label_prob)
    class_id = np.argmax(label_prob, axis=1)
    print(class_id)


if __name__ == '__main__':
    main()
