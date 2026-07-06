from kozax.genetic_programming import GeneticProgramming
from kozax.fitness_functions.Gymnax_fitness_function import GymFitnessFunction

import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np

# Graphics and plotting.
import mediapy as media
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)


# @title Import MuJoCo, MJX, and Brax
from datetime import datetime
import functools
import os
import time
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
from IPython.display import HTML, clear_output
import jax
import jax.numpy as jnp
import jax.random as jr
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp

print("These device(s) are detected:", jax.devices())

# Functions

from mujoco_playground import registry
from mujoco_playground._src.dm_control_suite.swimmer import Swim

class MujocoGymnaxWrapper:

    def __init__(self, env_name = None, config_overrides = None):
        if env_name == "SwimmerSwim":
            self.env = Swim(n_links=3)
        elif config_overrides is not None:
            self.env = registry.load(env_name, config_overrides=config_overrides)
        else:
            self.env = registry.load(env_name)
        # self.observation_space = self.env.observation_size
        self.action_space = self.env.action_size

    # state, env_state = self.env.reset(subkey, self.env_params)

    def reset(self, key, params=None):
        #state consists of pipeline_state, obs, reward, done, metrics, info
        state = self.env.reset(key)
        return state.obs, state

     # next_state, next_env_state, reward, done, _ = self.env.step(
     #            subkey, env_state, action, self.env_params
     #        )

    def step(self, key, state, action, params=None):

      action = action.reshape(self.action_space)
      next_state = self.env.step(state, action)

      done = next_state.done
      obs = next_state.obs
      reward = next_state.reward

      return obs, next_state, reward, done, {}

    def render(self, rollout):
      return self.env.render(rollout)

    @property
    def dt(self):
        return self.env.dt

def compute_trajectory_py(key, env, policy, T=1000, stride=1):
  jit_reset = jax.jit(env.reset)
  jit_step = jax.jit(env.step)
  obs, env_state = jit_reset(key)
  states = []

  for t in range(T):
      action = policy(obs)
      obs, env_state, reward, done, _ = jit_step(key, env_state, action)

      if t % stride == 0:
          states.append(env_state)

      if done:
          break

  return states


def run_environment(env, num_outputs, seed = 0, layer_sizes = 2, pop_size = 100, num_pop = 5, num_gen = 100, batch_size = 16, max_nodes=31):
    mujoco_env = MujocoGymnaxWrapper(env, config_overrides={'njmax': 500, 'naconmax': 2_000_000})

    fitness_function = GymFitnessFunction.__new__(GymFitnessFunction)
    fitness_function.env = mujoco_env
    fitness_function.env_params = None
    fitness_function.num_steps = 1000

    #Define operators and variables
    operator_list = [
        ("+", lambda x, y: jnp.add(x, y), 2, 0.5),
        ("-", lambda x, y: jnp.subtract(x, y), 2, 0.1),
        ("*", lambda x, y: jnp.multiply(x, y), 2, 0.5),
        ("sin", lambda x: jnp.sin(x), 1, 0.1),
        ("cos", lambda x: jnp.cos(x), 1, 0.1),
    ]

    obs, _ = fitness_function.env.reset(jax.random.PRNGKey(0))
    variable_list = [[f"y{i}" for i in range(obs.shape[0])]]

    #Initialize strategy
    strategy = GeneticProgramming(num_gen, pop_size, fitness_function, operator_list, variable_list, num_populations=num_pop, device_type = 'gpu', layer_sizes=jnp.array([layer_sizes]), num_outputs_per_tree=num_outputs, max_nodes = max_nodes)

    key = jr.PRNGKey(seed)
    data_key, gp_key = jr.split(key, 2)

    # The data comprises keys need to initialize the batch of environments.
    batch_keys = jr.split(data_key, batch_size)

    strategy.fit(gp_key, batch_keys, verbose=1)

if __name__ == '__main__':

    envs = ['WalkerWalk']#['ReacherEasy', 'SwimmerSwim', 'HopperHop', 'WalkerWalk', 'CheetahRun']#['CartpoleBalance', 'ReacherEasy', 'SwimmerSwim', 'HopperHop', 'WalkerWalk', 'CheetahRun']
    output_sizes = [3]
    layout_sizes = [2]
    pop_sizes = [50,100,200,500,1000]
    seeds = [0, 26, 10]

    for i, env in enumerate(envs):
        for pop_size in pop_sizes:
            for seed in seeds:                
                print("Env:" + env + ", seed:" + str(seed) + ". " + str(layout_sizes[i]) +" tree with "+ str(output_sizes[i]) + " outputs. Popsize: "+ str(pop_size))
                start = time.time()
                run_environment(env, output_sizes[i], seed, layer_sizes=layout_sizes[i], pop_size = pop_size) 
                end = time.time()
                print(f"Total runtime of Env: {env}, seed: {seed} is {end - start} seconds")
                print()
