import os

import gym
from multiprocessing import Pool
from gym.envs.mujoco import humanoid
# import mujoco_py


def main(e):
    # env = humanoid.HumanoidEnv()
    env = gym.make('Humanoid-v2')
    # obs = env.reset()

    # for e in range(50):
    env.reset()
    for _ in range(10000):
        obs, r, d, info = env.step(env.action_space.sample())
        # env.render()
        # print(r)
        if d:
            print(e, ' done')
            break

    env.close()


if __name__ == '__main__':
    __file__ = 'main.py'
    cpu_count = os.cpu_count() - 2
    pool = Pool(cpu_count)
    op = [pool.apply_async(main, args=(e,)) for e in range(50)]

