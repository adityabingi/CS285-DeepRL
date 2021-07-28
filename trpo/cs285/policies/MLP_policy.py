import abc
import itertools
from torch import nn
from torch._six import inf
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

        if self.discrete:
            action_distribution = distributions.Categorical(logits = self.logits_na(observation))
        else:
            mean = self.mean_net(observation)
            logstd = self.logstd.expand_as(mean)
            std = torch.exp(logstd)
            action_distribution = distributions.Normal(loc=mean, scale=std)

        return action_distribution

    def logprobs(self, observations, actions):

        if self.discrete:
            return self(observations).log_prob(actions)

        return self(observations).log_prob(actions).sum(dim=1)


#####################################################
#####################################################

class TRPOPolicy(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size,
                 cg_steps, damping, max_backtracks, max_kl,  **kwargs):

        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.baseline_loss = nn.MSELoss()

        self.cg_steps = cg_steps
        self.cg_damping = damping
        self.max_backtracks = max_backtracks
        self.max_kl = max_kl

    #################################

    def policy_parameters(self):
        if self.discrete:
            return self.logits_na.parameters()

        return itertools.chain(self.mean_net.parameters(), [self.logstd])

    def update(self, observations, actions, advantages, q_values=None):
        """
           TRPO policy update fucntion

        """
        self.observations = ptu.from_numpy(observations)
        self.actions = ptu.from_numpy(actions)
        self.advantages = ptu.from_numpy(advantages)

        # computes the loss that should be optimized when training with policy gradient

        log_probs = self.logprobs(self.observations, self.actions)

        with torch.no_grad():
            old_log_probs = self.logprobs(self.observations, self.actions)
        
        loss = self.surrogate_reward(log_probs, old_log_probs)

        # find policy gradient with surrogate objective of TRPO
        grads = torch.autograd.grad(loss, self.policy_parameters())
        policy_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        step_dir = self.conjugate_gradient(-policy_grad)

        max_step = torch.sqrt(2 *self.max_kl/torch.dot(step_dir, self.fisher_vector_product(step_dir)))
        full_step = max_step * step_dir
        expected_improve = torch.dot(-policy_grad, full_step)

        prev_params =  ptu.flatten_params(self.policy_parameters()).clone()
        success, new_params = self.line_search(old_log_probs, prev_params, full_step, expected_improve)
        ptu.assign_params_to(self.policy_parameters(), new_params)


        return loss.item()

    def run_baseline_prediction(self, obs):
        """
            Helper function that converts `obs` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `obs`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        obs = ptu.from_numpy(obs)
        predictions = self.baseline(obs)
        return ptu.to_numpy(predictions)[:, 0]

    def surrogate_reward(self, logprobs, oldlogprobs):
        """
           surrogate objective function for TRPO
        """
        return -(self.advantages * torch.exp(logprobs-oldlogprobs)).mean()

    def conjugate_gradient(self, b, residual_tol=1e-10):
        """
           Conjugate gradient descent algorithm

           For Conjugate gradient descent algorithm and derivation refer below link

           http://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/conjugate_direction_methods.pdf
        """
        x_k = torch.zeros(b.size())
        d_k = b.clone()
        g_k = b.clone()
        g_dot_g = torch.dot(g_k, g_k)


        for _ in range(self.cg_steps):

            fvp = self.fisher_vector_product(d_k)
            alpha = g_dot_g / torch.dot(d_k, fvp)
            x_k += alpha * d_k

            g_k -= alpha * fvp 
            new_g_dot_g = torch.dot(g_k, g_k)

            beta = new_g_dot_g / g_dot_g
            d_k = g_k + beta * d_k
            g_dot_g = new_g_dot_g

            if g_dot_g < residual_tol:
                break

        return x_k


    def line_search(self, oldlogprobs, prev_params, fullstep, expected_improve, accept_ratio=0.1):
        """
           line search to find optimal parameters in trust region
        """

        logprobs = self.logprobs(self.observations, self.actions).detach()
        prev_surr_reward = self.surrogate_reward(logprobs, oldlogprobs)

        for stepfrac in [.5**x for x in range(self.max_backtracks)]:
            new_params = prev_params + stepfrac * fullstep
            ptu.assign_params_to(self.policy_parameters(), new_params)

            logprobs = self.logprobs(self.observations, self.actions).detach()
            surr_reward = self.surrogate_reward(logprobs, oldlogprobs)
            improved = prev_surr_reward - surr_reward
            expected_improve = expected_improve * stepfrac
            ratio = improved/expected_improve

            if ratio.item() > accept_ratio:
                return True, new_params

        return False, prev_params


    def fisher_vector_product(self, vector):
        """
           Helper_fn to compute Hessian vector product to be used in cg algorithm
        """
        dist_old = self(self.observations)
        dist_new = self(self.observations)
        kl_loss = self.kl_divergence(dist_new, dist_old)

        grads = torch.autograd.grad(kl_loss, self.policy_parameters(), create_graph=True)
        grad_vector = torch.cat([grad.view(-1) for grad in grads])
        grad_vector_product = torch.sum(grad_vector * vector)
        grad_grads = torch.autograd.grad(grad_vector_product, self.policy_parameters())
        fisher_vector_product = torch.cat([grad.contiguous().view(-1) for grad in grad_grads]).detach()

        return fisher_vector_product + self.cg_damping * vector 

    def kl_divergence(self, dist_p, dist_q):

        """ Kl-divergence between two policy distributions 

            For continuous action space kl-div between multivariate normals is implemenred as given here
            https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence
        """

        if self.discrete:

            # kl-divergence between two categorical distributions
            t = dist_p.probs * (dist_p.logits - dist_q.logits)
            t[(dist_q.probs == 0).expand_as(t)] = inf
            t[(dist_p.probs == 0).expand_as(t)] = 0
            return t.sum(-1).mean()

        mean_old, std_old = dist_p.mean.detach(), dist_p.stddev.detach()
        mean, std = dist_q.mean, dist_q.stddev
        # start to calculate the kl-divergence
        kl = -torch.log(std / std_old) + (std.pow(2) + (mean - mean_old).pow(2)) / (2 * std_old.pow(2)) - 0.5
        kl_f =  kl.sum(-1, keepdim=True)

        return kl_f.mean()
