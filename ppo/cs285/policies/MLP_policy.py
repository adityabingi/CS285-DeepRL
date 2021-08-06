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
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers, size=self.size)
            self.logstd = nn.Parameter(
                torch.zeros((1,self.ac_dim), dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain(self.mean_net.parameters(), [self.logstd]),
                self.learning_rate
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################
    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO: get this from hw1
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes

        if self.discrete:
            action = self(ptu.from_numpy(observation)).sample()
        else:
            action = self(ptu.from_numpy(observation)).rsample()

        return ptu.to_numpy(action)

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
            mean = self.mean_net(observation)
            logstd = self.logstd.expand_as(mean)
            std = torch.exp(logstd)
            action_distribution = distributions.Normal(loc=mean, scale=std)

        return action_distribution

    def logprobs(self, observations, actions):

        if len(observations.shape) > 1:
            observations = observations
        else:
            observations = observations[None]

        if isinstance(observations, np.ndarray):
            observations = ptu.from_numpy(observations)

        if isinstance(actions, np.ndarray):
            actions = ptu.from_numpy(actions)

        if self.discrete:
            return self(observations).log_prob(actions)

        return self(observations).log_prob(actions).sum(dim=-1)


#####################################################
#####################################################

class PPOPolicy(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size,
                 clip_eps, ent_coeff, max_grad_norm,  **kwargs):

        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.baseline_loss = nn.MSELoss()

        self.clip_eps = clip_eps
        self.ent_coeff = ent_coeff
        self.max_grad_norm = max_grad_norm

    #################################

    def policy_parameters(self):
        if self.discrete:
            return self.logits_na.parameters()

        return itertools.chain(self.mean_net.parameters(), [self.logstd])

    def update(self, observations, actions, advantages, old_log_probs, q_values=None):
        self.observations = ptu.from_numpy(observations)
        self.actions = ptu.from_numpy(actions)
        self.advantages = ptu.from_numpy(advantages)
        old_log_probs = ptu.from_numpy(old_log_probs).detach()

        # computes the loss that should be optimized when training with policy gradient

        log_probs = self.logprobs(self.observations, self.actions)

        loss = self.ppo_surrogate_reward(log_probs, old_log_probs)

        ent_bonus = self(self.observations).entropy().mean()
        loss -= self.ent_coeff * ent_bonus

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item()


    def ppo_surrogate_reward(self, logprobs, oldlogprobs):
        """ 
           ppo objective function
        """
        ratio = torch.exp(logprobs-oldlogprobs)
        surr1 = (ratio * self.advantages)
        surr2 = (torch.clamp(ratio, 1.0-self.clip_eps, 1.0+self.clip_eps) * self.advantages)
        loss = -torch.min(surr1, surr2).mean()

        return loss
