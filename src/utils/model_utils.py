import random
from collections import deque

import torch



class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.memory.append((state, action, torch.FloatTensor([reward]), torch.FloatTensor([next_state])))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)