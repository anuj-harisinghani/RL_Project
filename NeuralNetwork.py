from tensorflow.keras import models, layers, initializers
'''
extra info:

remove tensorflow gpu
neural network with less layers but more neurons performs faster. 
'''


def random_kernel_for_layer(mean, stddev):
    return initializers.RandomNormal(mean=mean, stddev=stddev)


class NeuralNetwork:
    def __init__(self, n_obs, n_actions, n_hidden, mean, stddev):
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.mean = mean
        self.stddev = stddev

    def create_model_random(self):
        model = models.Sequential()

        # input layer
        model.add(layers.Dense(self.n_hidden, input_dim=self.n_obs, activation='sigmoid',
                               kernel_initializer=random_kernel_for_layer(self.mean, self.stddev)))
        # hidden layer
        # model.add(layers.Dense(n_hidden, activation='sigmoid', kernel_initializer=random_kernel_for_layer(mean, stddev)))
        # output layer
        model.add(layers.Dense(self.n_actions, activation='softmax',
                               kernel_initializer=random_kernel_for_layer(self.mean, self.stddev)))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

        # print(model.summary())
        return model

    def create_model(self):
        model = models.Sequential()

        # input layer
        model.add(layers.Dense(self.n_hidden, input_dim=self.n_obs, activation='sigmoid'))
        # hidden layer
        # model.add(layers.Dense(n_hidden, activation='sigmoid', kernel_initializer=random_kernel_for_layer(mean, stddev)))
        # output layer
        model.add(layers.Dense(self.n_actions, activation='softmax'))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

        print(model.summary())
        return model
