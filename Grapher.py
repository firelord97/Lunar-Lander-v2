import pandas as pd
import matplotlib.pyplot as plt
import os
import re

base_dir = '7642Spring2024achattarji3/Project_2'

filenames_single = ['evaluation_model_best_tau_0.0005_lr_0.0005_gamma_0.99_avg_238.csv']
# filenames_multiple = ['DQN_agent_tau_0.001_lr_0.001_gamma_0.99_batch_256.csv',
#                       'DQN_agent_tau_0.001_lr_0.0005_gamma_0.99_batch_256.csv',
#                       'DQN_agent_tau_0.001_lr_0.0003_gamma_0.99_batch_256.csv',
#                       'DQN_agent_tau_0.001_lr_0.0001_gamma_0.99_batch_256.csv']

def plot_single_file(filename):
    data_path = os.path.join(base_dir, 'data', filename)
    data = pd.read_csv(data_path)
    
    plt.figure(figsize=(8, 6))
    plt.plot(data['Episode'], data['Score'], label='Score')
    plt.plot(data['Episode'], data['Average_Score'], label='Moving Average Score', linestyle='--')
    plt.axhline(y=200, color='r', linestyle='--', label='Goal Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Scores and Moving Average Scores')
    plt.legend()
    graph_path = os.path.join(base_dir, 'graphs', f'{filename[:-4]}.png')
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

def plot_multiple_files(filenames):
    plt.figure(figsize=(8, 6))
    for filename in filenames:
        data_path = os.path.join(base_dir, 'data', filename)
        data = pd.read_csv(data_path)
        match = re.search(r'tau_(0.\d+)_lr_(0.\d+)_gamma_(0.\d+)', filename)
        if match:
            tau, lr, _ = match.groups()
            label = f"tau={tau}, lr={lr}"
            plt.plot(data['Episode'], data['Average_Score'], label=label)
    plt.axhline(y=200, color='r', linestyle='--', label='Goal Score')
    plt.xlabel('Episode')
    plt.ylabel('Moving Average Score')
    plt.title('Comparison of Average Scores')
    plt.legend()
    graph_path = os.path.join(base_dir, 'graphs', 'tau_0.001.png')
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

def main():
    graphs_dir = os.path.join(base_dir, 'graphs')
    if not os.path.exists(graphs_dir):
        os.makedirs(graphs_dir)

    for filename in filenames_single:
        plot_single_file(filename)

    # if filenames_multiple:
    #     plot_multiple_files(filenames_multiple)

if __name__ == "__main__":
    main()
