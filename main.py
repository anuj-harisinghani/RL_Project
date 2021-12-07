import os
import numpy as np
import multiprocessing as mp
import pickle
from tqdm import tqdm

from map_elites import MapElites
from map_elites import Individual


n_hidden = 256
modes = ['default', 'nov_mutation', 'nov_mutation', 'nov_mutation', 'nov_mutation', 'nov_mutation']
dist_range = [None, 1, 2, 3, 4]


# params for MAPElites
# mode = 'default' if dist == None else 'novelty_mutation'
n_behaviors = 2
# n_niches = 20
bootstrap_archive = None
bootstrap_genome_map = None
map_iterations = 1000
# n_init_niches = 50

# default params for Individual
fit_generations = 10

n_niches = 25
n_cells = n_niches ** n_behaviors
n_init_niches = int(0.4*n_cells)


def main(dist, mode, boot_arch, boot_gm):

    # actual running
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    default_map = MapElites(mode=mode, map_iterations=map_iterations, n_niches=n_niches, dist_threshold=dist,
                            n_init_niches=n_init_niches, n_hidden=n_hidden,
                            bootstrap_archive=boot_arch, bootstrap_genome_map=boot_gm)
    default_map.map_algorithm()

    ar = default_map.archive
    gm = default_map.genome_map

    network_map = np.empty_like(gm)
    for r in range(len(gm)):
        for c in range(len(gm[r])):
            if gm[r][c] != None:
                network_map[r][c] = gm[r][c].genome


    indices = np.argwhere(ar != 0)
    max_fit_ind = np.argwhere(ar == np.max(ar))


    print_dist = dist
    if dist == None:
        print_dist = 0

    archsave = './archive/{}_{}_{}_{}_{}.pkl'.format(mode, int(print_dist*10), map_iterations, n_niches, n_hidden)

    with open(archsave, 'wb') as arch:
        pickle.dump(ar, arch)

    return 0



if __name__ == '__main__':
    from gym.envs.mujoco.humanoid import HumanoidEnv
    from NeuralNetwork import NeuralNetwork

    mean = np.random.uniform(-15, 15)
    stddev = np.random.uniform(-5, 5)
    n_actions = HumanoidEnv().action_space.shape[0]
    n_obs = HumanoidEnv().observation_space.shape[0]

    model = NeuralNetwork(n_obs, n_actions, n_hidden, mean, stddev).create_model(print_summary=True)

    # creating bootstrap archive - created with n_behaviours and n_niches
    burner_map = MapElites(map_iterations=map_iterations, n_niches=n_niches, n_init_niches=n_init_niches,
                           dist_threshold=None, n_hidden=n_hidden)
    # burner_map.init_archive()

    dist_threshold = None
    for i in tqdm(range(n_init_niches), desc='bootstrap_archive'):
        burner_x = Individual(fit_generations, dist_threshold, n_hidden)
        burner_x.init_random_genome()
        burner_x.fit_genome()
        burner_map.update_archive(burner_x)

    bootstrap_archive = burner_map.archive
    bootstrap_genome_map = burner_map.genome_map

    bootsave = './archive/{}_{}_{}_{}.pkl'.format('bootstrap', map_iterations, n_niches, n_hidden)
    with open(bootsave, 'wb') as bf:
        pickle.dump(bootstrap_archive, bf)

    # check number of initialized cells
    bootstrap_indices = np.argwhere(bootstrap_archive != 0)
    max_fit_ind = np.argwhere(bootstrap_archive == np.max(bootstrap_archive))

    # multiprocessing - for some reason it's not working
    # cpu_count = os.cpu_count()
    # pool = mp.Pool(processes=cpu_count)
    # cv = [pool.apply_async(main, args=(dist_range[i], modes[i], bootstrap_archive, bootstrap_genome_map))
    #       for i in range(len(dist_range))]
    # op = [p.get() for p in cv]

    for i in range(len(dist_range)):
        main(dist_range[i], modes[i], bootstrap_archive, bootstrap_genome_map)
