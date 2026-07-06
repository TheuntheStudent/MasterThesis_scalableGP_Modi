# %%
# Specify the cores to use for XLA
import functools
import os
import jax
import jax.numpy as jnp
import jax.random as jr
import time

#import gymnax
import brax
from brax import envs
import matplotlib.pyplot as plt

from kozax.genetic_programming import GeneticProgramming
from kozax.fitness_functions.Gymnax_fitness_function import GymFitnessFunction
from brax import envs

class BraxGymnaxWrapper:

    def __init__(self, env_name, backend):
        self.env = envs.get_environment(env_name=env_name, backend=backend)
        self.observation_space = self.env.observation_size
        self.action_space = self.env.action_size

    def reset(self, key, params=None):
        #state consists of pipeline_state, obs, reward, done, metrics, info
        state = self.env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params=None):

      next_state = self.env.step(state, action)

      done = next_state.done

      # Freeze state after done
      obs = jnp.where(done, state.obs, next_state.obs)
      new_state = jax.tree.map(
          lambda new, old: jnp.where(done, old, new),
          next_state,
          state
      )

      reward = jnp.where(done, 0.0, next_state.reward)

      return obs, new_state, reward, done, {}


def run_environment(env, num_outputs, seed = 0, backend = 'generalized', pop_size = 100, num_pop = 5, num_gen = 50, batch_size = 16, max_nodes=31):
    brax_env = BraxGymnaxWrapper(env, backend=backend)

    fitness_function = GymFitnessFunction.__new__(GymFitnessFunction)
    fitness_function.env = brax_env
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
    strategy = GeneticProgramming(num_gen, pop_size, fitness_function, operator_list, variable_list, num_populations=num_pop, device_type = 'gpu', layer_sizes = jnp.array([num_outputs]), max_nodes = max_nodes)

    key = jr.PRNGKey(seed)
    data_key, gp_key = jr.split(key, 2)

    # The data comprises keys need to initialize the batch of environments.
    batch_keys = jr.split(data_key, batch_size)

    strategy.fit(gp_key, batch_keys, verbose=1)

if __name__ == '__main__':

    envs_list = ['inverted_pendulum', 'inverted_double_pendulum', 'reacher', 'swimmer', 'hopper', 'walker2d', 'halfcheetah', 'ant']
    output_sizes = [1, 1, 2, 2, 3, 6, 6, 8]
    seeds = [0, 26, 10]
    max_nodes = [31, 31, 31, 31, 31, 7, 7, 7]

    for i, env in enumerate(envs_list):
        for seed in seeds:
            print("Env:" + env + ", seed:" + str(seed))
            start = time.time()
            run_environment(env, output_sizes[i], seed, max_nodes = max_nodes[i]) 
            end = time.time()
            print(f"Total runtime of Env: {env}, seed: {seed} is {end - start} seconds")
            print()


