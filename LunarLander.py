import gym
import pandas as pd
import numpy as np
import os
from DQNAgent import DQNAgent

script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

model_save_dir = os.path.join(script_dir, 'models')
data_save_dir = os.path.join(script_dir, 'data')

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
if not os.path.exists(data_save_dir):
    os.makedirs(data_save_dir)

env = gym.make('LunarLander-v2')
n_games = 1000
batch_size = 256

# Hyperparameters to test
tau_values = [0.005, 0.001, 0.0005]
learning_rates = [0.001, 0.0005, 0.0003, 0.0001]
gamma_values = [0.99, 0.95]

# Iterating over the combinations of hyperparameters
for tau in tau_values:
    for optimizer_alpha in learning_rates:
        for gamma in gamma_values:
            # Initialize agent with current set of hyperparameters
            agent = DQNAgent(tau=tau, gamma=gamma, epsilon=1.0, optimizer_alpha=optimizer_alpha,
                             n_states=8, batch_size=batch_size, n_actions=4)
            scores = []
            avg_scores = []
            highest_avg_score = 200  # Initialize with the minimum acceptable average score

            for i in range(n_games):
                score = 0
                done = False
                observation = env.reset()
                while not done:
                    action = agent.get_action(observation)
                    observation_, reward, done, info = env.step(action)
                    score += reward
                    agent.push_action(observation, action, reward, observation_, done)
                    agent.learn()
                    observation = observation_
                scores.append(score)
                avg_score = np.mean(scores[-100:])
                avg_scores.append(avg_score)

                # Save the model if the current avg score is the highest and above 200
                if avg_score > highest_avg_score:
                    highest_avg_score = avg_score  # Update the highest recorded average score
                    model_name = f'model_best_tau_{tau}_lr_{optimizer_alpha}_gamma_{gamma}_avg_{int(highest_avg_score)}.pth'
                    save_path = os.path.join(model_save_dir, model_name)
                    agent.save_model(save_path)
                    print(f'New best model saved to {save_path} with avg score of {avg_score:.2f} at episode {i}')

                if i % 10 == 0:
                    print(f'Tau {tau}, learning rate {optimizer_alpha}, gamma {gamma}')
                    print(f'Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}')

            # Save performance data to a DataFrame and then to a CSV file
            df = pd.DataFrame({
                'Episode': range(1, n_games + 1),
                'Score': scores,
                'Average_Score': avg_scores
            })
            file_name = os.path.join(data_save_dir, f'DQN_agent_tau_{tau}_lr_{optimizer_alpha}_gamma_{gamma}_batch_{batch_size}.csv')
            df.to_csv(file_name, index=False)
            print(f'Data saved to {file_name}')
