from kozax.genetic_programming import GeneticProgramming
from kozax.fitness_functions.Gymnax_fitness_function import GymFitnessFunction

import jax
import jax.numpy as jnp
import jax.random as jr

import brax
from brax import envs
import matplotlib.pyplot as plt

class BraxGymnaxWrapper:
    def __init__(self, env_name, backend):
        self.env = envs.get_environment(env_name=env_name, backend=backend)
        self.observation_space = self.env.observation_size
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

      # Freeze state after done
      obs = jnp.where(done, state.obs, next_state.obs)
      new_state = jax.tree.map(
          lambda new, old: jnp.where(done, old, new),
          next_state,
          state
      )

      reward = jnp.where(done, 0.0, next_state.reward)

      return obs, new_state, reward, done, {}

brax_env = BraxGymnaxWrapper('ant', 'generalized')
key = jr.PRNGKey(0)

#Define hyperparameters
population_size = 100
num_populations = 5
num_generations = 25
batch_size = 16

fitness_function = GymFitnessFunction.__new__(GymFitnessFunction)
fitness_function.env = brax_env
#in Gymnax, env_params are max_speed: float = 8.0, max_torque: float = 2.0, dt: float = 0.05, g: float = 10.0, m: float = 1.0, l: float = 1.0, max_steps_in_episode: int = 200
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
strategy = GeneticProgramming(num_generations, population_size, fitness_function, operator_list, variable_list, num_populations=num_populations, device_type = 'gpu', num_outputs_per_tree=8, max_nodes = 15)

key = jr.PRNGKey(0)
data_key, gp_key = jr.split(key, 2)

# The data comprises keys need to initialize the batch of environments.
batch_keys = jr.split(data_key, batch_size)

strategy.fit(gp_key, batch_keys, verbose=1)

