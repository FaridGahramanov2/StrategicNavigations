import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(file_path, episode):
    q_table = np.load(file_path)
    plt.figure(figsize=(12, 10))  # Increase figure size
    ax = sns.heatmap(q_table, cmap='viridis', linewidths=.5)
    plt.title(f'Q-Table Heatmap at Episode {episode}', fontsize=14)
    plt.xlabel('Actions', fontsize=12)
    plt.ylabel('States', fontsize=12)
    plt.show()

# Example of plotting heatmaps for the Q-tables
episodes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]  # Adjust as necessary
for episode in episodes:
    file_path = f'q_table_{episode}.npy'
    plot_heatmap(file_path, episode)
