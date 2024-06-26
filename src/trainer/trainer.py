import gc
import gym
import sys
import time
import math
import random
import imageio
from itertools import count
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributed as dist
from torch.optim.lr_scheduler import OneCycleLR

from tools import TrainingLogger
from trainer.build import get_model, get_data_loader
from utils import RANK, LOGGER, colorstr, init_seeds
from utils.filesys_utils import *
from utils.model_utils import ReplayMemory




class Trainer:
    def __init__(
            self, 
            config,
            mode: str,
            device,
            is_ddp=False,
            resume_path=None,
        ):
        init_seeds(config.seed + 1 + RANK, config.deterministic)

        # init
        self.mode = mode
        self.is_training_mode = self.mode in ['train', 'resume']
        self.device = torch.device(device)
        self.is_ddp = is_ddp
        self.is_rank_zero = True if not self.is_ddp or (self.is_ddp and device == 0) else False
        self.config = config
        self.world_size = len(self.config.device) if self.is_ddp else 1
        if self.is_training_mode:
            self.save_dir = make_project_dir(self.config, self.is_rank_zero)
            self.wdir = self.save_dir / 'weights'

        # path, data params
        self.config.is_rank_zero = self.is_rank_zero
        self.resume_path = resume_path

        # environment define
        self.env = gym.make('CartPole-v1')
        self.case_num = self.env.action_space.n
        self.config.case_num = self.case_num

        # init model and training logger
        self.q_net, self.target = self._init_model(self.config, self.mode)
        self.training_logger = TrainingLogger(self.config, self.is_training_mode)

        # save the yaml config
        if self.is_rank_zero and self.is_training_mode:
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
            final_div_factor = self.lr / 25 / 5e-5
            self.scheduler = OneCycleLR(self.optimizer, max_lr=self.lr, total_steps=total_steps, pct_start=pct_start, final_div_factor=final_div_factor)


    def _init_model(self, config, mode):
        def _resume_model(resume_path, device, is_rank_zero):
            try:
                checkpoints = torch.load(resume_path, map_location=device)
            except RuntimeError:
                LOGGER.warning(colorstr('yellow', 'cannot be loaded to MPS, loaded to CPU'))
                checkpoints = torch.load(resume_path, map_location=torch.device('cpu'))
            q_net.load_state_dict(checkpoints['q_net'])
            target.load_state_dict(checkpoints['target'])
            del checkpoints
            torch.cuda.empty_cache()
            gc.collect()
            if is_rank_zero:
                LOGGER.info(f'Resumed model: {colorstr(resume_path)}')
            return q_net, target

        # init model and tokenizer
        do_resume = mode == 'resume' or (mode == 'validation' and self.resume_path)
        q_net, target = get_model(config, self.device)

        # resume model or resume model after applying peft
        if do_resume:
            q_net, target = _resume_model(self.resume_path, self.device, config.is_rank_zero)

        # init ddp
        if self.is_ddp:
            torch.nn.parallel.DistributedDataParallel(q_net, device_ids=[self.device])
            torch.nn.parallel.DistributedDataParallel(target, device_ids=[self.device])
        
        return q_net, target


    def do_train(self):
        self.train_cur_step = -1
        self.train_time_start = time.time()
        
        if self.is_rank_zero:
            LOGGER.info(f'\nUsing {self.dataloaders["train"].num_workers * (self.world_size or 1)} dataloader workers\n'
                        f"Logging results to {colorstr('bold', self.save_dir)}\n"
                        f'Starting training for {self.epochs} epochs...\n')
        
        if self.is_ddp:
            dist.barrier()

        for epoch in range(self.epochs):
            start = time.time()
            self.epoch = epoch

            if self.is_rank_zero:
                LOGGER.info('-'*100)

            for phase in self.modes:
                if self.is_rank_zero:
                    LOGGER.info('Phase: {}'.format(phase))

                if phase == 'train':
                    self.epoch_train(phase, epoch)
                    if self.is_ddp:
                        dist.barrier()
                else:
                    self.epoch_validate(phase, epoch)
                    if self.is_ddp:
                        dist.barrier()
            
            # clears GPU vRAM at end of epoch, can help with out of memory errors
            torch.cuda.empty_cache()
            gc.collect()

            # Early Stopping
            if self.is_ddp:  # if DDP training
                broadcast_list = [self.stop if self.is_rank_zero else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if not self.is_rank_zero:
                    self.stop = broadcast_list[0]
            
            if self.stop:
                break  # must break all DDP ranks

            if self.is_rank_zero:
                LOGGER.info(f"\nepoch {epoch+1} time: {time.time() - start} s\n\n\n")

        if RANK in (-1, 0) and self.is_rank_zero:
            LOGGER.info(f'\n{epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            

    def epoch_train(
            self,
            phase: str,
            epoch: int
        ):
        self.model.train()
        train_loader = self.dataloaders[phase]
        nb = len(train_loader)

        if self.is_ddp:
            train_loader.sampler.set_epoch(epoch)

        # init progress bar
        if RANK in (-1, 0):
            logging_header = ['CE Loss', 'Accuracy', 'lr']
            pbar = init_progress_bar(train_loader, self.is_rank_zero, logging_header, nb)

        for i, (x, label, attn_mask) in pbar:
            # Warmup
            self.train_cur_step += 1
            if self.train_cur_step <= self.warmup_steps_n:
                self.optimizer.param_groups[0]['lr'] = lr_warmup(self.train_cur_step, self.warmup_steps_n, self.lr0, self.lf)
            cur_lr = self.optimizer.param_groups[0]['lr']
            
            batch_size = x.size(0)
            x, label, attn_mask = x.to(self.device), label.to(self.device), attn_mask.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x, attn_mask)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            train_acc = torch.sum(torch.argmax(output, dim=-1).detach().cpu() == label.detach().cpu()) / batch_size

            if self.is_rank_zero:
                self.training_logger.update(
                    phase, 
                    epoch + 1,
                    self.train_cur_step,
                    batch_size, 
                    **{'train_loss': loss.item(), 'lr': cur_lr},
                    **{'train_acc': train_acc.item()}
                )
                loss_log = [loss.item(), train_acc.item(), cur_lr]
                msg = tuple([f'{epoch + 1}/{self.epochs}'] + loss_log)
                pbar.set_description(('%15s' * 1 + '%15.4g' * len(loss_log)) % msg)
            
        # upadate logs
        if self.is_rank_zero:
            self.training_logger.update_phase_end(phase, printing=True)
        
        
    def epoch_validate(
            self,
            phase: str,
            epoch: int,
            is_training_now=True
        ):
        def _init_log_data_for_vis():
            data4vis = {'x': [], 'y': [], 'pred': []}
            return data4vis

        def _append_data_for_vis(**kwargs):
            for k, v in kwargs.items():
                self.data4vis[k].append(v)

        with torch.no_grad():
            if self.is_rank_zero:
                if not is_training_now:
                    self.data4vis = _init_log_data_for_vis()

                val_loader = self.dataloaders[phase]
                nb = len(val_loader)
                logging_header = ['CE Loss', 'Accuracy']
                pbar = init_progress_bar(val_loader, self.is_rank_zero, logging_header, nb)

                self.model.eval()

                for i, (x, label, attn_mask) in pbar:
                    batch_size = x.size(0)
                    x, label, attn_mask = x.to(self.device), label.to(self.device), attn_mask.to(self.device)

                    output = self.model(x, attn_mask)
                    loss = self.criterion(output, label)

                    output = torch.argmax(output, dim=-1)
                    val_acc = torch.sum(output.detach().cpu() == label.detach().cpu()) / batch_size

                    self.training_logger.update(
                        phase, 
                        epoch, 
                        self.train_cur_step if is_training_now else 0, 
                        batch_size, 
                        **{'validation_loss': loss.item()},
                        **{'validation_acc': val_acc.item()}
                    )

                    loss_log = [loss.item(), val_acc.item()]
                    msg = tuple([f'{epoch + 1}/{self.epochs}'] + loss_log)
                    pbar.set_description(('%15s' * 1 + '%15.4g' * len(loss_log)) % msg)

                    if not is_training_now:
                        _append_data_for_vis(
                            **{'x': x.detach().cpu(),
                               'y': label.detach().cpu(),
                               'pred': output.detach().cpu()}
                        )

                # upadate logs and save model
                self.training_logger.update_phase_end(phase, printing=True)
                if is_training_now:
                    self.training_logger.save_model(self.wdir, self.model)
                    self.training_logger.save_logs(self.save_dir)

                    high_fitness = self.training_logger.model_manager.best_higher
                    low_fitness = self.training_logger.model_manager.best_lower
                    self.stop = self.stopper(epoch + 1, high=high_fitness, low=low_fitness)


    def vis_statistics(self, phase):
        # validation
        self.epoch_validate(phase, 0, False)
        all_x = torch.cat(self.data4vis['x'], dim=0)
        all_y = torch.cat(self.data4vis['y'], dim=0)
        all_pred = torch.cat(self.data4vis['pred'], dim=0)

        # cal statistics
        class_name = ['negative', 'mediocre', 'positive']
        print(classification_report(all_y, all_pred, target_names=class_name))
        
        # visualization the entire statistics
        vis_save_dir = os.path.join(self.config.save_dir, 'vis_outputs') 
        os.makedirs(vis_save_dir, exist_ok=True)

        cm = confusion_matrix(all_y, all_pred)
        cm = pd.DataFrame(cm, index=class_name, columns=class_name)
        plt.figure(figsize=(12, 10))
        plt.imshow(cm, cmap='Blues', interpolation=None)
        plt.title('Visualized Statistics', fontsize=20)
        plt.xlabel('Prediction', fontsize=20)
        plt.ylabel('True', fontsize=20)
        plt.xticks(range(len(class_name)), class_name, fontsize=17, rotation=30)
        plt.yticks(range(len(class_name)), class_name, fontsize=17, rotation=30)
        plt.colorbar()
        
        for i in range(cm.shape[1]):
            for j in range(cm.shape[0]):
                plt.text(i, j, round(cm.iloc[j, i], 1), ha='center', va='center', fontsize=17)
        
        plt.savefig(os.path.join(vis_save_dir, 'statistics.png'))


    def print_prediction_results(self, phase, result_num):
        if result_num > len(self.dataloaders[phase].dataset):
            LOGGER.info(colorstr('red', 'The number of results that you want to see are larger than total test set'))
            sys.exit()

        # validation
        self.epoch_validate(phase, 0, False)
        all_x = torch.cat(self.data4vis['x'], dim=0)
        all_y = torch.cat(self.data4vis['y'], dim=0)
        all_pred = torch.cat(self.data4vis['pred'], dim=0)

        ids = random.sample(range(all_x.size(0)), result_num)
        all_x = all_x[ids]
        all_y = all_y[ids]
        all_pred = all_pred[ids]

        for x, y, p in zip(all_x, all_y, all_pred):
            gt_label, pred_label = label_mapping(y), label_mapping(p)
            LOGGER.info(colorstr(self.tokenizer.decode(x.tolist())))
            LOGGER.info('*'*100)
            LOGGER.info(f'gt  : {gt_label}')
            LOGGER.info(f'pred: {pred_label}')
            LOGGER.info('*'*100 + '\n'*2)