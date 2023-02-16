import gym
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os

os.environ['SDL_VIDEODRIVER'] = 'dummy'

device = torch.device('cuda:0')
EPISODES = 500
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
GAMMA = 0.8
LR = 0.001
BATCH_SIZE = 64

class DQNAgent:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        ).to(device)

        self.target = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        ).to(device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.model.parameters(), LR)
        self.steps_done = 0
        self.memory = deque(maxlen=10000)

    def memorize(self, state, action, reward, next_state):
        self.memory.append((state, action, torch.FloatTensor([reward]), torch.FloatTensor([next_state])))


    def act(self, state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if random.random() > eps_threshold:
            return self.model(state).data.max(1)[1].view(1, 1).to(device)
        else:
            return torch.LongTensor([[random.randrange(2)]]).to(device)
    
    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        
        states = torch.cat(states).to(device)
        actions = torch.cat(actions).to(device)
        rewards = torch.cat(rewards).to(device)
        next_states = torch.cat(next_states).to(device)

        current_q = self.model(states).gather(1, actions)
        max_next_q = self.target(next_states).detach().max(1)[0]
        expected_q = rewards + (GAMMA * max_next_q)

        loss = F.mse_loss(current_q.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



env = gym.make('CartPole-v1')
agent = DQNAgent()
score_history = []



for e in range(1, EPISODES+1):
    state = env.reset()
    steps = 0

    while True:
        # env.render()
        state = torch.FloatTensor([state]).to(device)

        action = agent.act(state)

        next_state, reward, done, _ = env.step(action.item())

        if done:
            reward = -1
        
        agent.memorize(state, action, reward, next_state)
        agent.learn()

        state = next_state
        steps += 1

        if done:
            if e % 10 == 0:
                print('Episode {}: {}'.format(e, steps))
            break
    
    if e % 10 == 0:
        agent.target.load_state_dict(agent.model.state_dict())