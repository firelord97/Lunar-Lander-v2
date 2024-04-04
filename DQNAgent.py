import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ReplayBuffer(object):
    def __init__(self, states, batch_size, buffer_size) -> None:
        self.memory_size = buffer_size
        self.batch_size = batch_size
        self.count = 0
        self.states = np.zeros((self.memory_size, states), dtype=np.float32)
        self.actions = np.zeros(self.memory_size, dtype=np.int64)
        self.rewards = np.zeros(self.memory_size, dtype=np.float32)
        self.next_states = np.zeros((self.memory_size, states), dtype=np.float32)
        self.terminals = np.zeros(self.memory_size, dtype=np.uint8)

    def push_action(self, state, action, reward, next_state, terminal):
        if self.count >= self.memory_size:
            self.count = 0
        self.states[self.count] = state
        self.actions[self.count] = action
        self.rewards[self.count] = reward
        self.next_states[self.count] = next_state
        self.terminals[self.count] = terminal
        self.count += 1
        

    def sample_buffer(self):
        if (min(self.count, self.batch_size) < self.batch_size):
            return None
        batch = np.random.choice(min(self.count, self.memory_size), self.batch_size, replace=False)
        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        next_states = self.next_states[batch]
        terminal = self.terminals[batch]
        return states, actions, rewards, next_states, terminal
    
class DQN(nn.Module):
    def __init__(self, n_states, n_actions, optimizer_alpha) -> None:
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=optimizer_alpha)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        l1 = F.relu(self.fc1(state))
        l2 = F.relu(self.fc2(l1))
        action = F.relu(self.fc3(l1))

        return action
    
class DQNAgent:
    def __init__(self, tau, gamma, epsilon, optimizer_alpha, n_states, n_actions, batch_size, buffer_size=128000, epsilon_decay=0.9999, epsilon_min=0.03):
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_states = n_states
        self.n_actions = n_actions
        self.optimizer_alpha = optimizer_alpha
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.Q = DQN(n_states=self.n_states, n_actions=self.n_actions, optimizer_alpha=self.optimizer_alpha)
        self.Q_target = DQN(n_states=self.n_states, n_actions=self.n_actions, optimizer_alpha=self.optimizer_alpha)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q_target.eval()  # Set target network to evaluation mode
        self.memory = ReplayBuffer(states=n_states, buffer_size=buffer_size, batch_size=batch_size)
        self.count = 0

    def push_action(self, state, action, reward, next_state, terminal):
        self.memory.push_action(state, action, reward, next_state, terminal)
        self.count += 1

    def save_model(self, file_path):
        torch.save(self.Q.state_dict(), file_path)

    def get_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            state = torch.tensor(np.array(observation), dtype=torch.float32).to(self.Q.device)
            action = self.Q(state.unsqueeze(0)).max(1)[1].view(1, 1).item()
        return action

    def learn(self):
        if self.count < self.batch_size:
            return
        
        self.Q.optimizer.zero_grad()
        samples = self.memory.sample_buffer()
        if samples is None:
            return
        state, action, reward, next_state, terminal = samples
        state = torch.tensor(state).to(self.Q.device)
        action = torch.tensor(action).to(self.Q.device, dtype=torch.int64)
        reward = torch.tensor(reward).to(self.Q.device)
        next_state = torch.tensor(next_state).to(self.Q.device)
        terminal = torch.tensor(terminal).to(self.Q.device, dtype=torch.bool)

        # Get current Q estimates
        q_values = self.Q(state)
        q_value = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        # Double Q-Learning: Use Q to select actions and Q_target to evaluate
        next_actions = self.Q(next_state).max(1)[1].unsqueeze(1)
        next_q_values = self.Q_target(next_state).gather(1, next_actions).squeeze(-1)
        next_q_values[terminal] = 0.0

        # Compute the target Q values
        q_target = reward + self.gamma * next_q_values
        
        # Calculate loss
        loss = F.mse_loss(q_value, q_target)
        loss.backward()
        self.Q.optimizer.step()
        
        # Epsilon decay
        self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
        self.soft_update()


    def soft_update(self):
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)