import torch.nn as nn



class DQN(nn.Module):
    def __init__(self, config, device):
        super(DQN, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.case_num = config.case_num
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, case_num)
        )


    def forward(self, x):
        x = x.to(self.device)
        x = self.model(x)
        return x