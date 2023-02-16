import torch
import numpy as np


"""
common utils
"""

def save_checkpoint(file, model, optimizer):
    state = {'q_net': model[0].state_dict(), 'target': model[1].state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, file)
    print('model pt file is being saved\n')