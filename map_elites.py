import random

import numpy as np
import os
import gym

from NeuralNetwork import create_model_random, create_model_random_2, create_model, create_model_2

env = gym.make('Humanoid-v2')
n_actions = env.action_space.shape[0]
n_obs = env.observation_space.shape[0]
make_one_action = False

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class MapElites:
    def __init__(self, n_behaviors=2, n_niches=100, arch_shape=None,
                 map_iterations=100, n_init_niches=25, dist_threshold=0.5,
                 fit_generations=50):

        """
        :param n_behaviors: number of behaviors to track - defines dimensions of archive
        :param n_niches: the granularity of archive - this and n_behaviors will make a square archive
        :param arch_shape: (optional) only to specify shape of archive directly (if square archive is not required)
        :param map_iterations: number of iterations of main map elites algorithm
        :param n_init_niches: number of niches to randomly initialize, after which niches will be found by mutation
        :param dist_threshold: the threshold within which neighbors will be found (for custom mutation)
        :param fit_generations: number of generations to fit each genome for
        """

        # archive variables
        self.n_behaviors = n_behaviors
        self.n_niches = n_niches
        self.arch_shape = arch_shape
        self.archive = None
        self.genome_map = None

        # map elites algorithm variables
        self.map_iterations = map_iterations
        self.n_init_niches = n_init_niches
        self.dist_threshold = dist_threshold

        # genome variables
        self.fit_generations = fit_generations

        # initialize archive with zeros and the given dimensions
        self.init_archive()

    # initialize archive with zeros and
    # create genome map that keeps genome in cell (b1, b2) with b1, b2 being indices from the corresponding cell in archive
    def init_archive(self):
        if self.arch_shape is None:
            # create an archive with the given arch_dims and arch_size
            self.arch_shape = tuple(self.n_behaviors*[self.n_niches])

        # self.archive = np.random.random(shape)
        self.archive = np.zeros(self.arch_shape)
        self.genome_map = np.empty(shape=self.archive.shape, dtype='object')

    # generate a random solution (network/genome)
    def generate_random_solution(self):
        return Individual(self.fit_generations).init_random_genome()

    # randomly choose a non-empty cell from the archive
    def random_selection_from_archive(self):
        non_empty_indices = np.argwhere(self.archive != 0)
        r, c = random.choice(non_empty_indices)

        return r, c

    # default MAP Elites algorithm
    def default_algorithm(self):
        for i in range(self.map_iterations):

            # generate random solution if i < n_init_niches
            if i < self.n_init_niches:
                x = Individual(fit_generations=self.fit_generations, dist_threshold=None)
                x.init_random_genome()

            # else, select randomly from the archive and mutate
            else:
                # get the archive indices of the randomly selected individual
                r, c = self.random_selection_from_archive()
                x = self.genome_map[r][c]  # get the actual genome that was stored in those indices
                x = x.mutate_genome(self.arch_shape, r, c)  # mutate the genome

            # get behavior metric value and performance from fit_genome
            # behavior_indices = x.get_behavior()
            fitness = x.fit_genome()

    # MAP Elites algorithm with Novelty-based mutation
    def novelty_based_algorithm(self):
        for i in range(self.map_iterations):

            # generate random solution if i < n_init_niches
            if i < self.n_init_niches:
                x = Individual(fit_generations=self.fit_generations, dist_threshold=self.dist_threshold)
                x.init_random_genome()

            # else, select randomly from the archive and mutate
            else:
                # get the archive indices of the randomly selected individual
                r, c = self.random_selection_from_archive()
                x = self.genome_map[r][c]  # get the actual genome that was stored in those indices
                x = x.mutate_genome(self.arch_shape, r, c)  # mutate the genome

            # get behavior metric value and performance from fit_genome
            # behavior_indices = x.get_behavior()
            fitness = x.fit_genome()


class Individual:
    """
    Class Individual - makes each genome an object
    """
    def __init__(self, fit_generations, dist_threshold):
        self.generations = fit_generations
        self.dist_threshold = dist_threshold

        self.fitness = None
        self.genome = None

        self.mean = np.random.uniform(-2, 2)
        self.stddev = np.random.uniform(-1, 1)
        self.n_actions = n_actions
        self.n_obs = n_obs

    # randomly initialize a genome / network
    def init_random_genome(self):
        self.genome = create_model_random_2(self.n_obs, self.n_actions, self.mean, self.stddev)

    # function to fit the genome and produce total fitness score after specified number of generations
    def fit_genome(self):
        """
        :return: fitness score after given generations
        """

        # call the mujoco environment and run the genome g in it
        genome_fitness = []
        simulation = gym.make('Humanoid-v2')

        for g in range(self.generations):
            obs = simulation.reset()
            generation_reward = []
            while True:  # for _ in range(100000):
                preds = self.genome.predict(obs.reshape(-1, len(obs)))

                if make_one_action:
                    # preds is a (1, 17) shape vector, choose one action based on softmax
                    action = np.zeros(n_actions)
                    action[preds.argmax()] = 1

                else:
                    action = preds

                # step using the predicted action vector
                # simulation.render()
                # action = simulation.action_space.sample()
                obs, reward, done, info = simulation.step(action)
                generation_reward.append(reward)

                if done:
                    print(g, 'done')
                    genome_fitness.append(generation_reward)
                    break

        simulation.close()
        self.fitness = np.sum(genome_fitness)
        return genome_fitness

    # mutation function
    def mutate_genome(self, arch_shape, r, c):
        """
        :return: the mutated genome
        """

        if self.dist_threshold is None:
            k = 1
        else:
            k = self.get_n_neighbors(arch_shape, r, c)

        weights = self.genome.get_weights()

        # adding perturbations based on the number of neighbors found within the distance threshold to each layer separately
        for i in range(len(weights)):
            weights[i] += k * np.random.uniform(-1, 1, weights[i].shape)

        self.genome.set_weights(weights)
        return self.genome

    # get number of neighbors that sit in a given distance threshold
    # with help from this answer on StackOverflow: https://stackoverflow.com/a/44874588
    def get_n_neighbors(self, arch_shape, r, c):
        y, x = np.ogrid[:arch_shape[0], :arch_shape[1]]

        # get euclidean distances of all elements in the archive
        dist_from_given_indices = np.sqrt((x - c)**2 + (y - r)**2)

        # create mask on archive that shows True if a cell is within the distance threshold, and False if it isn't
        mask = dist_from_given_indices <= self.dist_threshold

        # k will be the number of cells found in the mask that are True (i.e., within the distance threshold)
        k = len(np.argwhere(mask == True))
        return k





