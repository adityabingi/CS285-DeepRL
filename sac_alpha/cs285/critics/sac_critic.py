from .base_critic import BaseCritic

import torch
from torch import nn
from torch import optim

from cs285.infrastructure import pytorch_util as ptu

def init_method(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0.0)

class SACCritic(nn.Module, BaseCritic):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime.
        is None
    """
    def __init__(self, hparams):
        super().__init__()
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate_valuefn']

        # critic parameters
        self.num_target_updates = hparams['num_target_updates']
        self.gamma = hparams['gamma']
        self.l2_reg = hparams['l2_reg']
        self.critic_network = ptu.build_mlp(
            self.ob_dim+self.ac_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
            init_method = init_method,
            activation = 'relu'
        )
        self.critic_network.to(ptu.device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.critic_network.parameters(),
            self.learning_rate,
        )

        self.apply(init_method)

    def forward(self, obs, acs):
        #obs = ptu.from_numpy(obs)
        #acs = ptu.from_numpy(acs)
        obs_acs = torch.cat([obs, acs], dim=-1)
        return self.critic_network(obs_acs).squeeze(1)

    def forward_np(self, obs, acs):
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        obs_acs = torch.cat([obs, acs], dim=-1)
        predictions = self(obs_acs)
        return ptu.to_numpy(predictions)

    def update(self, ob_no, ac_no, targets):
        
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                targets: shape: (sum_of_path_lengths,)

            returns:
                training loss
        """
    
        targets = targets.detach()

        for _ in range(self.num_target_updates):

            rand_indices = torch.randperm(targets.shape[0])
            q_values = self(ob_no, ac_no)[rand_indices]
            q_targets = targets[rand_indices]

            total_loss = self.loss(q_values, q_targets)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return total_loss.item()
