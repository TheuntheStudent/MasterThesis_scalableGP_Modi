import time
from typing import Dict

# Specify the cores to use for XLA
import functools
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=10'
import jax
import jax.numpy as jnp
import jax.random as jr

#import gymnax
import brax
from brax import envs
import matplotlib.pyplot as plt
from brax.envs.wrappers import EpisodeWrapper

def step_fn(policy):

    def _step(carry, t):
        env_state = carry
        action = jnp.asarray(policy(env_state.obs))
        jax.debug.print("action={action}", action = action)
        new_state = jax.jit(env.step)(env_state, action)
        obs = new_state.obs
        reward = new_state.reward

        jax.debug.print("obs={obs}", obs = obs[7])
        jax.debug.print("action_calc={actions}", actions = jnp.tanh(jnp.array([obs[7] - (obs[8] * obs[5]), obs[6] + obs[8]])))
        jax.debug.print("reward={reward}", reward = reward)
        jax.debug.print("")

        return new_state, (obs, reward, action)

    return _step

def visualize_trajectory(all_obs, treward, actions):
  # Convert to arrays
  all_actions = jnp.array(actions)
  all_rewards = jnp.array(treward)
  reward = jnp.sum(all_rewards)
  all_obs = jnp.array(all_obs)
  all_list = []
  plot_list = []

  for i in range(1000):
    theta1 = jnp.arctan2(all_obs[i, 1], all_obs[i, 0])
    theta2 = jnp.arctan2(all_obs[i, 3], all_obs[i, 2])

    x1 = 0.1 * jnp.cos(theta1)
    y1 = 0.1 * jnp.sin(theta1)

    x2 = x1 + 0.1 * jnp.cos(theta1 + theta2)
    y2 = y1 + 0.1 * jnp.sin(theta1 + theta2)

    all_list.append([x2, y2])

    if i % 50 == 0:

        print(all_obs[i, 5], all_obs[i, 6], all_obs[i, 7], all_obs[i, 8])
        print(all_obs[i, 7] - (all_obs[i, 8] * all_obs[i, 5]), all_obs[i, 6] + all_obs[i, 8])
        print(jnp.tanh(all_obs[i, 7] - (all_obs[i, 8] * all_obs[i, 5])), jnp.tanh(all_obs[i, 6] + all_obs[i, 8]))

        print(
                f"Timestep {i} : "
                f"x-coordinate tip: {x2} | "
                f"y-coordinate tip: {y2} | "
                f"Angular velocity joint 1: {all_obs[i, 6]} | "
                f"Angular velocity joint 2: {all_obs[i, 7]} | "
            )
        print()

        plot_list.append([i, x1, y1, x2, y2])

  nr_of_rows = 4
  nr_of_cols = 5

  fig, axs = plt.subplots(nrows=nr_of_rows, ncols=nr_of_cols, figsize=(12, 12))
  fig.suptitle("State evolution over time")
  for i in range(len(plot_list)):
    c = i % nr_of_cols
    r = int(i / nr_of_cols)
    t, x1, y1, x2, y2 = plot_list[i]
    axs[r, c].scatter(0,0, c = "black", label = "middle point")
    axs[r, c].scatter(x2, y2, c="blue", label = "arm tip")
    axs[r, c].plot([0, x1], [0, y1])
    axs[r, c].plot([x1, x2], [y1, y2])
    axs[r, c].scatter(all_obs[t, 4], all_obs[t, 5], c = "red", label = "target location")

    if i != 0:
      x2s_y2s = jnp.array(all_list[t-50:t])
      x2s = x2s_y2s[:, 0]
      y2s = x2s_y2s[:, 1]
      axs[r, c].plot(x2s, y2s, alpha = 0.4, label = "tip trajectory")

    axs[r, c].set_title("t="+ str(t))
    axs[r, c].set_xlim(-0.3, 0.3)
    axs[r, c].set_ylim(-0.3, 0.3)
    if r == 0 and c == nr_of_cols-1:
      axs[r, c].legend(loc='best', fontsize = 6)
    if i > (nr_of_rows-1) * nr_of_cols:
      axs[r, c].set_xlabel("x")
    if c == 0:
      axs[r, c].set_ylabel("y")

  plt.tight_layout()
  plt.show()

  print("Total reward:", reward)

  plt.plot(all_obs[:,8], alpha = 0.4, label='x-value diff')
  plt.plot(all_obs[:,9], alpha = 0.4, label='y-value diff')
  plt.plot(jnp.add(all_obs[:, 8], all_obs[:, 9]), label="total value diff")
  plt.plot(all_rewards, label="reward")
  plt.plot(all_actions, label="actions")
  plt.legend()
  plt.show()

def compute_and_visualize(obs, env_state, policy, T):

  # JIT the whole loop
  jax.debug.print("obs={obs}", obs = obs[7])
  jax.debug.print("actions={actions}", actions = jnp.tanh(jnp.array([obs[7] - (obs[8] * obs[5]), obs[6] + obs[8]])))
  jax.debug.print("reward={reward}", reward = 0)

  _, (all_obs, treward, actions) = jax.lax.scan(step_fn(policy), env_state, jnp.arange(T))

  visualize_trajectory(all_obs, treward, actions)

if __name__ == '__main__':
    T = 1000
    env = EpisodeWrapper(envs.get_environment("reacher"), episode_length=T, action_repeat=1)
    policy = lambda obs: jnp.tanh(jnp.array([obs[7]-(obs[8]*obs[5]), obs[6]+obs[8]])) #[y7-(y8*y5), y6+y8]
    key = jnp.array([1278412471, 2182328957], dtype=jnp.uint32)
    env_state = env.reset(key)
    print(env_state.obs, key)
    compute_and_visualize(env_state.obs, env_state, policy, T)