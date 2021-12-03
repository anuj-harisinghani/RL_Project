import os

import gym
import mujoco_py
from gym.envs.mujoco.humanoid import HumanoidEnv
from multiprocessing import Pool

env = HumanoidEnv()
obs = env.reset()


# def main(e):
#     # env = humanoid.HumanoidEnv()
#     env = gym.make('Humanoid-v2')
#     # obs = env.reset()
#
#     # for e in range(50):
#     env.reset()
#     for _ in range(10000):
#         obs, r, d, info = env.step(env.action_space.sample())
#         env.render()
#         # print(r)
#         if d:
#             print(' done')
#             break
#
#     env.close()


# if __name__ == '__main__':
#     __file__ = 'main.py'
#     cpu_count = os.cpu_count() - 2
#     pool = Pool(cpu_count)
#     op = [pool.apply_async(main, args=(e,)) for e in range(50)]

# ============================================================
'''
testing for number of nodes in hidden layer to see which is the fastest
do it when no other apps are running
'''
#
# import time
# import numpy as np
# time_taken = []
# from map_elites import Individual
# n_hiddens = [75]  # list(range(50, 150, 5))
# for n_hidden in n_hiddens:
#     last_time = time.time()
#     x = Individual(50, 5, n_hidden)
#     x.init_random_genome()
#     r = x.fit_genome()
#
#     time_taken.append(time.time() - last_time)
#
# print(min(time_taken), np.argmin(time_taken))

# =============================================================
import os

archive_path = os.path.join('archive')
filename_format = 'fitness_{}_{}.pkl'
genotype_format = 'genotype_{}_{}.pkl'

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from map_elites import Individual
import dill as pickle
from pathos.multiprocessing import Pool
import time
import numpy as np
from tqdm import tqdm

np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)


def fit(index):
    print('instance start', index)
    # lt = time.time()
    x = Individual(50, 5, 34)
    x.init_random_genome()
    r = x.fit_genome()
    genotype = x.genome
    # r = [0, 1]
    # genotype = [0, 0]
    fit_path = os.path.join(archive_path, filename_format.format(index, index))
    gen_path = os.path.join(archive_path, genotype_format.format(index, index))

    print('index reached here', index)

    with open(fit_path, 'wb') as fit_file:
        pickle.dump(r, fit_file)
    fit_file.close()

    with open(gen_path, 'wb') as gen_file:
        pickle.dump(genotype, gen_file)
    gen_file.close()


    # if os.path.exists(fit_path):
    #     with open(fit_path, 'rb') as r_fit_file:
    #         old_r = pickle.load(r_fit_file)
    #     r_fit_file.close()
    #
    #     old_val = np.sum(np.sum(old_r))
    #     new_val = np.sum(np.sum(r))
    #     if new_val > old_val:
    #         print('updating fitness', old_val, new_val)
    #         overwrite()
    # else:
    #     print('adding fitness first time')
    #     overwrite()

    # print('instance end', index, time.time() - lt)
    return index


results = []


def callback(val):
    results.append(val)


if __name__ == '__main__':
    lt = time.time()
    __file__ = 'main.py'
    cpu_count = os.cpu_count() - 2  # don't use all cpu cores
    pool = Pool(cpu_count)
    cv = [pool.apply_async(fit, args=(e,), callback=callback) for e in range(16)]
    pool.close()
    pool.join()

    print(time.time() - lt)


'''
N Behaviors = 2

1. Step length (metres) = {min: 0, max: 1.15 * (leg length)} [1] [2]
2. Speed (metres/sec) = {min: 0, max: 3.33333... * (leg length)} [3]

Sources for information:
[1] relation between stride length and leg length https://blog.issaonline.edu/stride-length-vs.-stride-rate
[2] relation between stride length and step length https://shorturl.at/nuxzL
[3] relation between speed and leg length https://www.researchgate.net/publication/12458273_Aspects_of_body_self-calibration#pf6 , page 284


leg length of humanoid bot (from humanoid.xml) = 
based on these values, range(step_length) = (0, 
'''