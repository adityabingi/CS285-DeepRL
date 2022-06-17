import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy
from cs285.infrastructure.utils import *

LOG_STD_MIN = -20
LOG_STD_MAX = 2

def init_method(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0.0)


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.common= ptu.build_mlp(input_size=self.ob_dim,
                                      output_size=self.size,
                                      n_layers=1, size=self.size,
                                      activation = 'relu',
                                      output_activation='relu')
            self.mean = nn.Linear(self.size, self.ac_dim)
            #self.logstd = nn.Linear(self.size, self.ac_dim)
            self.common.to(ptu.device)
            self.mean.to(ptu.device)
            #self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain(self.common.parameters(), self.mean.parameters()),
                self.learning_rate
            )

        self.apply(init_method)
    ##################################
    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: get this from hw1
        if self.discrete:
            action_distribution = distributions.Categorical(logits = self.logits_na(observation))
        else:
            out = self.common(observation)
            mean = self.mean(out)
            #logstd = self.logstd(out)
            #logstd = torch.clamp(logstd, LOG_STD_MIN, LOG_STD_MAX)
            #std = torch.exp(logstd)
            #action_distribution = distributions.Normal(loc=mean, scale=std)

        return mean


#####################################################
#####################################################

class TD3_Policy(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size,
                 action_low, action_high, **kwargs):

        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.baseline_loss = nn.MSELoss()
        self.action_scale = torch.FloatTensor((action_high -action_low)/2.0).to(ptu.device)
        self.action_bias  = torch.FloatTensor((action_high + action_low)/2.0).to(ptu.device)

    #################################

    def policy_parameters(self):
        if self.discrete:
            return self.logits_na.parameters()

        return itertools.chain(self.common.parameters(), self.mean.parameters())

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        raw_action = self(ptu.from_numpy(observation))
        squash_action = torch.tanh(raw_action)
        action = (squash_action * self.action_scale + self.action_bias)

        return ptu.to_numpy(action)[0]

    def get_action_batch(self, observations: torch.FloatTensor, eps=1e-6):

        if len(observations.shape) > 1:
            observations = observations
        else:
            observations = observations[None]

        if isinstance(observations, np.ndarray):
            observations = ptu.from_numpy(observations)

        raw_actions = self(observations)
        squash_actions = torch.tanh(raw_actions)
        actions = squash_actions * self.action_scale + self.action_bias

        return actions

    def update(self, observations, q_networks):
        #self.observations = ptu.from_numpy(observations)
        pass