import gym
import mujoco_py

from gym.envs.mujoco import humanoid

env = humanoid.HumanoidEnv()
obs = env.reset()

for i in range(10000):
    obs, r, d, info = env.step(env.action_space.sample())
    env.render()

env.close()