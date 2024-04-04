import gym
import pandas as pd
import numpy as np
import os
import torch
from DQNAgent import DQN

def evaluate_model(model_path, env, n_episodes=1000):
    # Load the model
    model = DQN(n_states=env.observation_space.shape[0], n_actions=env.action_space.n, optimizer_alpha=0.001)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode

    scores = []
    for i_episode in range(n_episodes):
        observation = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = model(torch.tensor(observation, dtype=torch.float32)).max(0)[1].view(1, 1).item()
            observation, reward, done, _ = env.step(action)
            total_reward += reward
        
        scores.append(total_reward)
        print(f'Episode {i_episode} finished with score: {total_reward}')

    return scores

def main():
    env = gym.make('LunarLander-v2')
    
    model_paths = ['7642Spring2024achattarji3/Project_2/models/model_best_tau_0.005_lr_0.001_gamma_0.99_avg_252.pth',
                   '7642Spring2024achattarji3/Project_2/models/model_best_tau_0.0005_lr_0.0005_gamma_0.99_avg_238.pth']
    
    for model_path in model_paths:
        scores = evaluate_model(model_path, env)
        avg_scores = [np.mean(scores[max(0, i-100):(i+1)]) for i in range(len(scores))]

        # Save scores to a DataFrame and then to a CSV file
        df = pd.DataFrame({
            'Episode': np.arange(len(scores)),
            'Score': scores,
            'Average_Score': avg_scores
        })
        filename = f'evaluation_{os.path.basename(model_path).replace(".pth", "")}.csv'
        df.to_csv(filename, index=False)
        print(f'Evaluation data saved to {filename}')

if __name__ == '__main__':
    main()
