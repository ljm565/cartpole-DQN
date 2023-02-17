import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

import gym
import math
import pickle
import random
import imageio
from tqdm import tqdm
from itertools import count
import matplotlib.pyplot as plt

from utils.config import Config
from models.dqn import DQN
from utils.utils_func import *
from utils.utils_model import ReplayMemory



class Trainer:
    def __init__(self, config:Config, device:torch.device, mode:str, continuous:int):
        self.config = config
        self.device = device
        self.mode = mode
        self.continuous = continuous

        # if continuous, load previous training info
        if self.continuous:
            with open(self.config.loss_data_path, 'rb') as f:
                self.loss_data = pickle.load(f)

        # path, data params
        self.base_path = self.config.base_path
        self.model_path = self.config.model_path
 
        # train params
        self.batch_size = self.config.batch_size
        self.episodes = self.config.episodes
        self.lr = self.config.lr

        # define memory class
        self.memory = ReplayMemory(10000)

        # environment define
        self.env = gym.make('CartPole-v1')
        self.case_num = self.env.action_space.n

        # model, optimizer, loss
        torch.manual_seed(999)  # for reproducibility
        self.q_net = DQN(self.config, self.case_num, self.device).to(self.device)
        self.target = DQN(self.config, self.case_num, self.device).to(self.device)
        self.target.load_state_dict(self.q_net.state_dict())
        self.target.eval()
        self.criterion = nn.SmoothL1Loss()
        if self.mode == 'train':
            total_steps = self.episodes
            pct_start = 10 / total_steps
            final_div_factor = self.lr / 25 / 5e-5
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
            self.scheduler = OneCycleLR(self.optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=pct_start, final_div_factor=final_div_factor)
            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.q_net.load_state_dict(self.check_point['q_net'])
                self.target.load_state_dict(self.check_point['target'])
                self.target.eval()
                self.optimizer.load_state_dict(self.check_point['optimizer'])
                del self.check_point
                torch.cuda.empty_cache()
        else:
            self.check_point = torch.load(self.model_path, map_location=self.device)
            self.q_net.load_state_dict(self.check_point['q_net'])
            self.target.load_state_dict(self.check_point['target'])
            self.q_net.eval()
            self.target.eval()
            del self.check_point
            torch.cuda.empty_cache()


    def select_action(self, state, phase='train'):
        if phase == 'test':
            with torch.no_grad():
                return torch.argmax(self.q_net(state), dim=1, keepdim=True)

        eps_threshold = self.config.eps_end + (self.config.eps_start - self.config.eps_end) * math.exp(-1. * self.steps_done / self.config.eps_decay)
        self.steps_done += 1

        # choose bigger action between left and right
        if random.random() > eps_threshold:
            with torch.no_grad():
                return torch.argmax(self.q_net(state), dim=1, keepdim=True)
        
        # random action between left and right
        return torch.tensor([[random.randrange(self.case_num)]], dtype=torch.long).to(self.device)
    

    def train(self):
        if len(self.memory) >= self.batch_size:
            batch = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states = zip(*batch)

            # batch data and current q
            states = torch.cat(states).to(self.device)
            actions = torch.cat(actions).to(self.device)
            rewards = torch.cat(rewards).to(self.device)
            next_states = torch.cat(next_states).to(self.device)

            # finding current q and max q values
            curr_q = torch.gather(self.q_net(states), dim=1, index=actions)
            max_next_q, _ = torch.max(self.target(next_states), dim=1)

            # target q 
            target_q = rewards + max_next_q * self.config.gamma
            target_q = target_q.detach()
            
            # training
            self.optimizer.zero_grad()
            loss = self.criterion(curr_q.squeeze(), target_q)
            loss.backward()
            self.optimizer.step()


    def training(self):
        self.steps_done = 0
        self.episode_duration = []
        self.best_duration = 0
        self.q_net.train()

        for episode in range(self.episodes):
            state = self.env.reset()
            
            for t in count():
                state = torch.FloatTensor([state])
                action = self.select_action(state)

                next_state, reward, done, _ = self.env.step(action.item())

                if done:
                    reward = -1
                
                # push to memory
                self.memory.push(state, action, reward, next_state)

                # update Q networks
                self.train()

                # update state
                state = next_state                

                if done:
                    self.episode_duration.append(t+1)
                    break
            
            self.scheduler.step()
                
            if episode % self.config.target_update_duration == 0:
                self.target.load_state_dict(self.q_net.state_dict())
                self.target.eval()

            if episode % 10 == 0:
                print('Episode {} duration: {}'.format(episode+1, self.episode_duration[-1]))

            if self.best_duration <= self.episode_duration[-1]:
                print('Episode {} duration: {}'.format(episode+1, self.episode_duration[-1]))
                self.best_duration = self.episode_duration[-1]
                save_checkpoint(self.model_path, [self.q_net, self.target], self.optimizer)

        print('Complete')
        self.env.render()
        self.env.close()

        return {'duration': self.episode_duration}
    

    def test(self):
        all_screens, durations = [], []
        episodes = 10
        for _ in tqdm(range(episodes)):
            screens = []
            state = self.env.reset()
            for t in count():
                screen = self.env.render('rgb_array')
                screens.append(screen)

                state = torch.FloatTensor([state])
                action = self.select_action(state, 'test')

                next_state, reward, done, _ = self.env.step(action.item())

                if done:
                    reward = -1
                
                # push to memory
                self.memory.push(state, action, reward, next_state)

                # update state
                state = next_state                

                if done:
                    durations.append(t+1)
                    break

            all_screens.append(screens)
        
        self.env.close()
        print('{} episode - average duration: {}, max duration: {}, min duration: {}'.format(episodes, sum(durations)/len(durations), max(durations), min(durations)))
        
        # make video
        selected_id = durations.index(max(durations))
        all_screens = all_screens[selected_id]
        model_name = self.model_path[self.model_path.rfind('/')+1:self.model_path.rfind('.')]
        save_p = self.base_path + 'results/' + model_name + '.gif'
        imageio.mimsave(save_p, all_screens, fps=50)