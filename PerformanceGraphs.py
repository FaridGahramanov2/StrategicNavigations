import matplotlib.pyplot as plt
import numpy as np  # This is useful if you need to load data from files or handle arrays

# Example: Load rewards data from numpy files or replace these lines with code to load your data
rewards_q_learning = np.load('rewards_q-learn.npy')  # Load rewards for Q-Learning
rewards_sarsa = np.load('rewards_sarsa.npy')

episodes = range(1, len(rewards_q_learning) + 1)  # Adjust based on the number of episodes data you have

plt.figure(figsize=(10, 5))
plt.plot(episodes, rewards_q_learning, label='Q-Learning', color='blue')
plt.plot(episodes, rewards_sarsa, label='SARSA', color='red')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Learning Curves')
plt.legend()
plt.grid(True)
plt.savefig('learning_curves.png')  # Save the figure to a file
plt.show()
