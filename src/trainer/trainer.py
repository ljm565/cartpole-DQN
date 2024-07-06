import gc
import gym
import math
import random
import imageio
from tqdm import tqdm
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from tools import TrainingLogger
from trainer.build import get_model
from utils import RANK, LOGGER, colorstr, init_seeds
from utils.filesys_utils import *
from utils.training_utils import *
from utils.model_utils import ReplayMemory




class Trainer:
    def __init__(
            self, 
            config,
            mode: str,
            device,
            resume_path=None,
        ):
        init_seeds(config.seed + 1 + RANK, config.deterministic)

        # init
        self.mode = mode
        self.is_training_mode = self.mode in ['train', 'resume']
        self.device = torch.device(device)
        self.config = config
        if self.is_training_mode:
            self.save_dir = make_project_dir(self.config)
            self.wdir = self.save_dir / 'weights'

        # path, data params
        self.resume_path = resume_path

        # environment define
        self.env = gym.make('CartPole-v1')
        self.case_num = self.env.action_space.n
        self.config.case_num = self.case_num

        # init model and training logger
        self.q_net, self.target = self._init_model(self.config, self.mode)
        self.training_logger = TrainingLogger(self.config, self.is_training_mode)

        # save the yaml config
        if self.is_training_mode:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.config.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', self.config)  # save run args

        # init replay memory
        self.memory = ReplayMemory(1000)

        # init criterion, optimizer, etc.
        self.batch_size = self.config.batch_size
        self.episodes = self.config.episodes
        self.criterion = nn.SmoothL1Loss()
        if self.is_training_mode:
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.lr)

            # init scheduler
            total_steps = self.episodes
            pct_start = 10 / total_steps
            final_div_factor = self.config.lr / 25 / 5e-5
            self.scheduler = OneCycleLR(self.optimizer, max_lr=self.config.lr, total_steps=total_steps, pct_start=pct_start, final_div_factor=final_div_factor)


    def _init_model(self, config, mode):
        def _resume_model(resume_path, device):
            try:
                checkpoints = torch.load(resume_path, map_location=device)
            except RuntimeError:
                LOGGER.warning(colorstr('yellow', 'cannot be loaded to MPS, loaded to CPU'))
                checkpoints = torch.load(resume_path, map_location=torch.device('cpu'))
            q_net.load_state_dict(checkpoints['model']['q_net'])
            target.load_state_dict(checkpoints['model']['target'])
            del checkpoints
            torch.cuda.empty_cache()
            gc.collect()
            LOGGER.info(f'Resumed model: {colorstr(resume_path)}')
            return q_net, target

        # init model and tokenizer
        do_resume = mode == 'resume' or (mode == 'validation' and self.resume_path)
        q_net, target = get_model(config, self.device)

        # resume model
        if do_resume:
            q_net, target = _resume_model(self.resume_path, self.device)

        return q_net, target
    

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


    def do_train(self):
        logging_header = ['Loss', 'duration', 'lr']
        pbar = init_progress_bar(self.episodes, logging_header)
        
        self.steps_done = 0
        self.train_cur_step = -1
        self.episode_duration = []
        self.best_duration = 0
        self.q_net.train()

        for episode in pbar:
            total_loss, total_batch = 0, 0
            state = self.env.reset()
            
            for t in count():
                self.train_cur_step += 1
                state = torch.FloatTensor([state])
                action = self.select_action(state)

                next_state, reward, done, _ = self.env.step(action.item())

                if done:
                    reward = -1
                
                # push to memory
                self.memory.push(state, action, reward, next_state)

                # update Q networks
                loss, batch_size = self.episode_train()
                total_loss += loss
                total_batch += batch_size

                # update state
                state = next_state                

                if done:
                    self.episode_duration.append(t+1)
                    break
            
            self.scheduler.step()
                
            if episode % self.config.target_update_duration == 0:
                self.target.load_state_dict(self.q_net.state_dict())
                self.target.eval()

            # upadate logs and save model
            self.training_logger.update(
                    'train', 
                    episode + 1,
                    self.train_cur_step,
                    self.batch_size, 
                    **{'lr': self.optimizer.param_groups[0]['lr']}
                )
            
            self.training_logger.update(
                    'validation', 
                    episode + 1,
                    self.train_cur_step,
                    self.batch_size, 
                    **{'loss': total_loss/total_batch if total_batch != 0 else 0},
                    **{'duration': self.episode_duration[-1]}
                )
            
            loss_log = [total_loss/total_batch if total_batch != 0 else 0, self.episode_duration[-1], self.optimizer.param_groups[0]['lr']]
            msg = tuple([f'{episode + 1}/{self.episodes}'] + loss_log)
            pbar.set_description(('%15s' * 1 + '%15.4g' * len(loss_log)) % msg)
            
            self.training_logger.update_phase_end('validation', printing=False)
            self.training_logger.save_model(self.wdir, {'q_net': self.q_net, 'target': self.target})
            self.training_logger.save_logs(self.save_dir)
            
            # if episode % 10 == 0:
            #     LOGGER.info('Episode {} duration: {}'.format(episode+1, self.episode_duration[-1]))

            # if self.best_duration <= self.episode_duration[-1]:
            #     print('Episode {} duration: {}'.format(episode+1, self.episode_duration[-1]))
            #     self.best_duration = self.episode_duration[-1]
            #     save_checkpoint(self.model_path, [self.q_net, self.target], self.optimizer)

        LOGGER.info('Training completed')
        self.env.render()
        self.env.close()


    def episode_train(self):
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
            return loss.item(), self.batch_size
        return 0, 0


    def make_gif(self):
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
        LOGGER.info(f'{episodes} episode - average duration: {sum(durations)/len(durations)}, max duration: {max(durations)}, min duration: {min(durations)}')
        
        # make video
        LOGGER.info('Video is generated..')
        selected_id = durations.index(max(durations))
        all_screens = all_screens[selected_id]
        vis_save_dir = os.path.join(self.config.save_dir, 'vis_outputs')
        os.makedirs(vis_save_dir, exist_ok=True)
        imageio.mimsave(os.path.join(vis_save_dir, 'cartpole.gif'), all_screens, fps=50)
