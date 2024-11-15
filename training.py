from connect_four import ConnectFour
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.FloatTensor([reward])
        action = torch.LongTensor([action])
        done = torch.FloatTensor([done])

        q_values = self.model(state)
        next_q_values = self.model(next_state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = reward + self.gamma * next_q_values.max(1)[0] * (1 - done)
        loss = self.criterion(q_value, next_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

env = ConnectFour()
state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
action_dim = env.action_space.n
agent1 = DQNAgent(state_dim, action_dim)
agent2 = DQNAgent(state_dim, action_dim)

num_episodes = 10000

for episode in range(num_episodes):
    state = env.reset().flatten()
    done = False
    total_reward1 = 0
    total_reward2 = 0

    while not done:
        if env.current_player == 1:
            action = agent1.choose_action(state)
        else:
            action = agent2.choose_action(state)

        next_state, reward, done, _ = env.step(action)
        next_state = next_state.flatten()

        if env.current_player == 1:
            agent1.learn(state, action, reward, next_state, done)
            total_reward1 += reward
        else:
            agent2.learn(state, action, reward, next_state, done)
            total_reward2 += reward

        state = next_state

    if episode % 1000 == 0:
        print(f"Episode {episode}, Total Reward Agent 1: {total_reward1}, Total Reward Agent 2: {total_reward2}")

print("Training completed.")