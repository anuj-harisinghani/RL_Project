import random

import numpy as np
import os
import math
import gym
from tqdm import tqdm
from gym.envs.mujoco.humanoid import HumanoidEnv, mass_center

from NeuralNetwork import NeuralNetwork

# getting environment variables
ENV = HumanoidEnv()
n_actions = ENV.action_space.shape[0]
n_obs = ENV.observation_space.shape[0]
make_one_action = False

# Forcing code to use CPU - GPU is slow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class MapElites:
    def __init__(self, n_behaviors=2, n_niches=100, arch_shape=None,
                 map_iterations=100, n_init_niches=25, dist_threshold=0.5,
                 fit_generations=50, n_hidden=75):

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
        self.n_hidden = n_hidden

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
        return Individual(self.fit_generations, self.dist_threshold, self.n_hidden).init_random_genome()

    # randomly choose a non-empty cell from the archive
    def random_selection_from_archive(self):
        non_empty_indices = np.argwhere(self.archive != 0)
        r, c = random.choice(non_empty_indices)

        return r, c

    # default MAP Elites algorithm
    def default_algorithm(self):
        for i in range(self.map_iterations):
            x = None
            # generate random solution if i < n_init_niches
            if i < self.n_init_niches:
                x = Individual(fit_generations=self.fit_generations, dist_threshold=None, n_hidden=self.n_hidden)
                x.init_random_genome()

            # else, select randomly from the archive and mutate
            else:
                # get the archive indices of the randomly selected individual
                r, c = self.random_selection_from_archive()
                x = self.genome_map[r][c]  # get the actual genome that was stored in those indices
                x = x.mutate_genome(self.arch_shape, r, c)  # mutate the genome

            # get behavior metric value and performance from fit_genome
            x.fit_genome()
            fitness = x.fitness
            step_dist = x.step_distance
            vel = x.velocity

    # MAP Elites algorithm with Novelty-based mutation
    def novelty_based_algorithm(self):
        for i in range(self.map_iterations):
            x = None
            # generate random solution if i < n_init_niches
            if i < self.n_init_niches:
                x = Individual(fit_generations=self.fit_generations, dist_threshold=self.dist_threshold, n_hidden=self.n_hidden)
                x.init_random_genome()

            # else, select randomly from the archive and mutate
            else:
                # get the archive indices of the randomly selected individual
                r, c = self.random_selection_from_archive()
                x = self.genome_map[r][c]  # get the actual genome that was stored in those indices
                x = x.mutate_genome(self.arch_shape, r, c)  # mutate the genome

            # get behavior metric value and performance from fit_genome
            x.fit_genome()
            fitness = x.fitness
            step_dist = x.step_distance
            vel = x.velocity




class Individual:
    """
    Class Individual - makes each genome an object
    """
    def __init__(self, fit_generations, dist_threshold, n_hidden):
        self.generations = fit_generations
        self.dist_threshold = dist_threshold
        self.n_hidden = n_hidden

        self.fitness = None
        self.step_distance = None
        self.velocity = None
        self.genome = None

        self.mean = np.random.uniform(-2, 2)
        self.stddev = np.random.uniform(-1, 1)
        self.n_actions = n_actions
        self.n_obs = n_obs

    # randomly initialize a genome / network
    def init_random_genome(self):
        self.genome = NeuralNetwork(n_obs, n_actions, self.n_hidden, self.mean, self.stddev).create_model_random()

    # function to fit the genome and produce total fitness score after specified number of generations
    def fit_genome(self):
        """
        use this function to optimize the network by simulating it, use GA or ES or something
        P.S. NO NEED TO OPTIMIZE USING GA OR ES - JUST LET THE GENOTYPE RUN AND RECORD ITS FITNESS AND BEHAVIOR METRICS
        THEN USE THOSE VALUES TO FILL THE ARCHIVE. LET THE MUTATION OF GENOTYPE CREATE NEW GENOTYPES AND IT WILL OPTIMIZE
        GENOTYPES FOUND IN A CELL IN THE ARCHIVE.
        :return: fitness score after given generations
        """

        # call the mujoco environment and run the genome g in it
        env = HumanoidEnv()
        sim = env.sim

        # fitness and behavior metrics - will be averaged across iterations, then averaged across generations
        genome_fitness = []
        step_distance = []
        velocity = []
        # speed metrics

        for g in tqdm(range(self.generations), desc='fitting genome'):
            obs = env.reset()
            generation_reward = []

            gen_step_distance = []  # step distance between contact.pos
            foot_timestep = 0
            foot_pos = np.array([0.0] * 3)

            gen_velocity = []
            center_of_mass = mass_center(env.model, sim)

            # simulation
            for iteration in range(100000):
                preds = self.genome.predict(obs.reshape(-1, len(obs)))

                if make_one_action:
                    # preds is a (1, 17) shape vector, choose one action based on softmax
                    action = np.zeros(n_actions)
                    action[preds.argmax()] = 1

                else:
                    action = preds

                # step using the predicted action vector
                obs, reward, done, info = env.step(action)
                generation_reward.append(reward)

                # behavior metric 1: step distance

                # calculate foot stepping distances
                for i in range(len(sim.data.contact)):
                    contact = sim.data.contact[i]

                    '''
                    contact ids:
                    floor = 0
                    left foot = 11
                    right foot = 8
                    '''

                    # initial filter - keeping contacts with the floor and removing all others
                    geom_list = [contact.geom1, contact.geom2]
                    if contact.geom1 != contact.geom2 and 0 in geom_list:

                        # check for any foot touching
                        if 11 in geom_list or 8 in geom_list:
                            dist_between_steps = math.dist(contact.pos, foot_pos)
                            # if iteration > foot_timestep and delta_distance_from_origin > 0:
                            if iteration > foot_timestep:
                                # print(geom_list)
                                # print('pos', contact.pos, foot_pos, dist_between_steps)

                                # append changes in step distances
                                gen_step_distance.append(dist_between_steps)

                                # update previous distance and position values
                                # dist = contact.dist
                                foot_pos = contact.pos
                                foot_timestep = iteration

                                # print('updated timesteps and position: ', foot_timestep, dist)

                # calculate speed - NOT using cvel (center of mass velocity), instead using mass_center from Humanoid
                new_center_of_mass = mass_center(env.model, sim)
                dt = env.model.opt.timestep * env.frame_skip
                v = (new_center_of_mass - center_of_mass) / dt
                gen_velocity.append(v)

                center_of_mass = new_center_of_mass

                # check end condition
                if done:
                    genome_fitness.append(np.mean(generation_reward))
                    step_distance.append(np.mean(gen_step_distance))
                    velocity.append(np.mean(gen_velocity))
                    break

        env.close()
        self.fitness = np.sum(genome_fitness)
        self.step_distance = np.mean(step_distance)
        self.velocity = np.mean(velocity)
        # return genome_fitness

    # mutation function
    def mutate_genome(self, arch_shape, r, c):
        """

        :param arch_shape: shape of the archive
        :param r: row index of individual selected in archive
        :param c: col index of individual selected in archive
        :return:
        """

        # make distinction between default map_elites and novelty_based map_elites
        if self.dist_threshold is None:
            k = 1
        else:
            k = self.get_n_neighbors(arch_shape, r, c)

        # adding perturbations based on the number of neighbors found within the distance threshold to each layer separately
        weights = self.genome.get_weights()
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
