"""
    Name: Farid
    Surname: Gahramanov
    Student ID: S023378
"""

import os.path
import numpy as np
from Environment import Environment
import time
import rl_agents

GRID_DIR = "grid_worlds/"

if __name__ == "__main__":
    file_name = input("Enter file name: ")

    assert os.path.exists(os.path.join(GRID_DIR, file_name)), "Invalid File"

    env = Environment(os.path.join(GRID_DIR, file_name))

    # Hyperparameters
    discount_rate = 0.95
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    alpha = 0.1
    max_episode = 500
    seed = 42

    # Initialize agents with parameters
    q_learning_agent = rl_agents.QLearningAgent(
        env=env,
        seed=seed,
        discount_rate=discount_rate,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        alpha=alpha,
        max_episode=max_episode
    )

    sarsa_agent = rl_agents.SARSAAgent(
        env=env,
        seed=seed,
        discount_rate=discount_rate,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        alpha=alpha,
        max_episode=max_episode
    )

    agents = [q_learning_agent, sarsa_agent]
    actions = ["UP", "LEFT", "DOWN", "RIGHT"]

    for agent in agents:
        print("*" * 50)
        print()

        env.reset()

        start_time = time.time_ns()

        agent.train()

        end_time = time.time_ns()

        path, score = agent.validate()

        print("Actions:", [actions[i] for i in path])
        print("Score:", score)
        print("Elapsed Time (ms):", (end_time - start_time) * 1e-6)

        print("*" * 50)
