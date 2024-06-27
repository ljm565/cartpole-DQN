import os

from models import DQN

PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders



def get_model(config, device):
    q_net = DQN(config, device).to(device)
    target = DQN(config, device).to(device)
    target.load_state_dict(q_net.state_dict())
    target.eval()
    return q_net, target