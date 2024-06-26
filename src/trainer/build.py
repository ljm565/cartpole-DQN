import os

import torch
from torch.utils.data import DataLoader, distributed, random_split

from models import DQN
from utils import LOGGER, RANK, colorstr

PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders



def get_model(config, device):
    q_net = DQN(config, device).to(device)
    target = DQN(config, device).to(device)
    target.load_state_dict(q_net.state_dict())
    target.eval()
    return q_net, target


def build_dataset(config, tokenizer, modes):
    def _init_data_size(data_len, train_prop):
        train_size = int(data_len * train_prop)
        val_size = (data_len - train_size) // 2
        test_size = data_len - train_size - val_size
        return train_size, val_size, test_size

    if config.google_store_review_train:
        data_path = preprocess_data(config.google_store_review.path)
        dataset = DLoader(config, read_dataset(data_path), tokenizer)
        train_l, val_l, test_l = _init_data_size(len(dataset), config.google_store_review.trainset_prop)
        trainset, valset, testset = random_split(dataset, [train_l, val_l, test_l])
        tmp_dsets = {'train': trainset, 'validation': valset, 'test': testset}
        dataset_dict = {mode: tmp_dsets[mode] for mode in modes}
    else:
        LOGGER.warning(colorstr('yellow', 'You have to implement data pre-processing code..'))
        raise NotImplementedError
    return dataset_dict


def build_dataloader(dataset, batch, workers, shuffle=True, is_ddp=False):
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers
    sampler = None if not is_ddp else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return DataLoader(dataset=dataset,
                              batch_size=batch,
                              shuffle=shuffle and sampler is None,
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              collate_fn=getattr(dataset, 'collate_fn', None),
                              worker_init_fn=seed_worker,
                              generator=generator)


def get_data_loader(config, tokenizer, modes, is_ddp=False):
    datasets = build_dataset(config, tokenizer, modes)
    dataloaders = {m: build_dataloader(datasets[m], 
                                       config.batch_size, 
                                       config.workers, 
                                       shuffle=(m == 'train'), 
                                       is_ddp=is_ddp) for m in modes}

    return dataloaders