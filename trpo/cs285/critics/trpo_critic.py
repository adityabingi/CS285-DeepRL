from .base_critic import BaseCritic

import torch
from torch import nn
from torch import optim

from cs285.infrastructure import pytorch_util as ptu


class TRPOCritic(nn.Module, BaseCritic):
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
        self.learning_rate = hparams['learning_rate']

        # critic parameters
        self.num_target_updates = hparams['num_target_updates']
        self.gamma = hparams['gamma']
        self.l2_reg =hparams['l2_reg']
        self.critic_network = ptu.build_mlp(
            self.ob_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.critic_network.to(ptu.device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.critic_network.parameters(),
            self.learning_rate,
        )

    def forward(self, obs):
        return self.critic_network(obs).squeeze(1)

    def forward_np(self, obs):
        obs = ptu.from_numpy(obs)
        predictions = self(obs)
        return ptu.to_numpy(predictions)

    def update(self, ob_no, targets):
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                targets: value function targets

            returns:
                value function loss
        """

        
        targets = ptu.from_numpy(targets).detach()

        for _ in range(self.num_target_updates):

            rand_indices = torch.randperm(targets.shape[0])
            v_ts = self(ptu.from_numpy(ob_no))[rand_indices]
            v_targets = targets[rand_indices]

            value_loss = self.loss(v_ts, v_targets)

            for param in self.critic_network.parameters():
                value_loss += param.pow(2).sum() * self.l2_reg

            self.optimizer.zero_grad()
            value_loss.backward()
            self.optimizer.step()

        return value_loss.item()
