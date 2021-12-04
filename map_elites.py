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

map_iter = 0


class MapElites:
    def __init__(self, mode='default',
                 n_behaviors=2, n_niches=20,
                 bootstrap_archive=None, bootstrap_genome_map=None,
                 map_iterations=1000, n_init_niches=50,
                 dist_threshold=None, fit_generations=10, n_hidden=35):

        """
        :param mode: mode of map_elites - can be 'default' or 'novelty-based'

        :param n_behaviors: number of behavior values - defines number of dimensions for archive
        :param n_niches: the granularity of archive - this and n_behaviors will make a square archive

        :param bootstrap_archive: an archive already initialized with n_init_niches
        :param bootstrap_genome_map: a genome map already initialized with n_init_niches

        :param map_iterations: number of iterations of main map elites algorithm
        :param n_init_niches: number of niches to randomly initialize, after which niches will be found by mutation

        :param dist_threshold: the threshold within which neighbors will be found (for custom mutation)
        :param fit_generations: number of generations to fit each genome for
        :param n_hidden: number of hidden nodes in the neural network
        """

        # archive variables
        self.n_behaviors = n_behaviors
        self.n_niches = n_niches
        self.bootstrap_archive = bootstrap_archive
        self.bootstrap_genome_map = bootstrap_genome_map
        self.arch_shape = None
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

    def init_archive(self):
        """
        Archive Intializer

        if provided with a bootstrap archive, make archive as the bootstrap_archive and go from there.
        else, initialize archive with zeros
        create genome map that keeps genome in cell (b1, b2) corresponding to the archive
        """
        if self.bootstrap_archive is not None:
            # use bootstrap
            self.archive = self.bootstrap_archive
            self.arch_shape = self.bootstrap_archive.shape
            self.genome_map = self.bootstrap_genome_map

        else:
            # create an empty archive with the given arch_dims and arch_size
            self.arch_shape = tuple(self.n_behaviors*[self.n_niches])
            self.archive = np.zeros(self.arch_shape)
            self.genome_map = np.empty(shape=self.archive.shape, dtype='object')

    def generate_random_solution(self):
        """
        generate a random solution (network/genome)
        :return: Individual object which is initalized randomly
        """

        return Individual(self.fit_generations, self.dist_threshold, self.n_hidden).init_random_genome()

    def random_selection_from_archive(self):
        """
        randomly choose a non-empty cell from the archive
        :return: row and col indices of the chosen genome
        """

        non_empty_indices = np.argwhere(self.archive != 0)
        r, c = random.choice(non_empty_indices)
        return r, c

    def update_archive(self, x):  # genome, fitness, step_dist, velocity):
        """
        updates the archive, given the fitness and behavior metrics
        """
        fitness = x.fitness
        step_dist = x.step_distance * 10  # had to increase this value since they were so small
        velocity = x.velocity

        max_step_dist = 1
        max_vel = 5

        step_dist_range = np.arange(0, max_step_dist, max_step_dist/self.n_niches)
        vel_range = np.arange(0, max_vel, max_vel/self.n_niches)

        # the archive is going to have step_distance as the rows and velocity as the columns
        row = np.argmin(np.abs(step_dist_range - step_dist))
        col = np.argmin(np.abs(vel_range - velocity))

        if fitness > self.archive[row][col]:
            self.archive[row][col] = fitness
            self.genome_map[row][col] = x

    def map_algorithm(self):
        """
        MAP Elites algorithm - modified to take in a pre-initialized bootstrap_archive and to incorporate
        the distance threshold feature for novelty_based MAP Elites algorithm
        """

        # if bootstrap_archive is not given, then bussiness as usual - initialize n_init_niches
        # if bootstrap_archive is given, then skip initializing n_init_niches
        if self.bootstrap_archive is None:
            start_index = 0
        else:
            start_index = self.n_init_niches

        for i in range(start_index, self.map_iterations):
            global map_iter
            map_iter = i

            x = None
            # generate random solution if i < n_init_niches
            if i < self.n_init_niches:
                x = Individual(self.fit_generations, self.dist_threshold, self.n_hidden)
                x.init_random_genome()

            # else, select randomly from the archive and mutate
            else:
                # get the archive indices of the randomly selected individual
                r, c = self.random_selection_from_archive()
                x = self.genome_map[r][c]  # get the actual genome that was stored in those indices
                x.mutate_genome(self.arch_shape, r, c)  # mutate the genome

            # get behavior metric value and performance from fit_genome
            x.fit_genome()
            self.update_archive(x)

    '''
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
        '''



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

        self.mean = np.random.uniform(-10, 10)
        self.stddev = np.random.uniform(-2, 2)
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

        for _ in tqdm(range(self.generations), desc=str(map_iter) + 'fitting genome'):
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
