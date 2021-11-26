import numpy as np
import os
import gym

env = gym.make('Humanoid-v2')
n_actions = env.action_space.shape[0]
n_obs = env.observation_space.shape[0]


class MapElites:
    def __init__(self, archive_shape):
        self.archive = self.random_init_archive(archive_shape)

    def default_algorithm(self):
        pass

    def novelty_based_algorithm(self):
        pass

    @staticmethod
    def random_init_archive(arch_dims=2, arch_size=100, arch_shape=None):
        if arch_shape is None:
            # create an archive with the given arch_dims and arch_size
            shape = tuple(arch_dims*[arch_size])
        else:
            # create an archive with the given shape
            shape = arch_shape
        empty_archive = np.zeros(shape)
        archive = np.random.random(shape)

        # incomplete - needs a way to only initialize G number of spaces in the archive

        return archive


class Population:
    def __init__(self, g, f, b):
        self.genome = g
        self.fitness = f
        self.behavior_features = b
        self.n_actions = n_actions
        self.n_obs = n_obs

    # @staticmethod
    def fit_genome(self, g, iterations):
        # call the mujoco environment and run the genome g in it
        env = gym.make('Humanoid-v2')
        env.reset()

        for _ in range(iterations):
            some_action = self.network()
            env.step(some_action)

    @staticmethod
    def mutate_genome(g, lr, k):
        """
        :param g: the genotype to mutate
        :param lr: learning rate alpha
        :param k: the number of neighbors found within a certain threshold of the individual we want to mutate
                  k will be None if it's the base algorithm of MAP Elites, otherwise it'll be an int
        :return: the mutated individual
        """

        if k is None:
            # base algorithm mutation - random
            pass
        else:
            # given an int value for k, make the mutation stronger with high k, and weak with low k
            pass

    @staticmethod
    def network():
        # neural network for taking in observations and predicting what action to take
        return 0
