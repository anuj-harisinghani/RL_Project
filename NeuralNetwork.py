import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import models, layers, initializers


def random_kernel_for_layer(mean, stddev):
    return initializers.RandomNormal(mean=mean, stddev=stddev)


def create_model_random(n_obs, n_actions, mean, stddev):
    n_hidden = 75
    model = models.Sequential()

    # input layer
    model.add(layers.Dense(n_obs, activation='sigmoid', kernel_initializer=random_kernel_for_layer(mean, stddev)))
    # hidden layer
    model.add(layers.Dense(n_hidden, activation='sigmoid', kernel_initializer=random_kernel_for_layer(mean, stddev)))
    # output layer
    model.add(layers.Dense(n_actions, activation='softmax', kernel_initializer=random_kernel_for_layer(mean, stddev)))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    print(model.summary())
    return model


def create_model_random_2(n_obs, n_actions, mean, stddev):
    n_hidden = 75
    model = models.Sequential()

    # input layer
    model.add(layers.Dense(n_hidden, input_dim=n_obs, activation='sigmoid', kernel_initializer=random_kernel_for_layer(mean, stddev)))
    # hidden layer
    # model.add(layers.Dense(n_hidden, activation='sigmoid', kernel_initializer=random_kernel_for_layer(mean, stddev)))
    # output layer
    model.add(layers.Dense(n_actions, activation='softmax', kernel_initializer=random_kernel_for_layer(mean, stddev)))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    print(model.summary())
    return model


def create_model(n_obs, n_actions):
    n_hidden = 75
    model = models.Sequential()

    # input layer
    model.add(layers.Dense(n_obs, input_dim=n_hidden, activation='sigmoid'))
    # hidden layer
    model.add(layers.Dense(n_hidden, activation='sigmoid'))
    # output layer
    model.add(layers.Dense(n_actions, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    print(model.summary())
    return model


def create_model_2(n_obs, n_actions):
    n_hidden = 75
    model = models.Sequential()

    # input layer
    model.add(layers.Dense(n_hidden, input_dim=n_obs, activation='sigmoid'))
    # hidden layer
    # model.add(layers.Dense(n_hidden, activation='sigmoid', kernel_initializer=random_kernel_for_layer(mean, stddev)))
    # output layer
    model.add(layers.Dense(n_actions, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    print(model.summary())
    return model
