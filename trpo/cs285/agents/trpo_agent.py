import numpy as np
from collections import OrderedDict
import torch

from .base_agent import BaseAgent
from cs285.policies.MLP_policy import TRPOPolicy
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.critics.trpo_critic import TRPOCritic

from cs285.infrastructure.utils import *

class TRPOAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(TRPOAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.use_gae = self.agent_params['use_gae']
        self.lam = self.agent_params['gae_lam']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        # actor/policy
        self.actor = TRPOPolicy(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['cg_steps'],
            self.agent_params['damping'],
            self.agent_params['max_backtracks'],
            self.agent_params['max_kl_increment'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
        )

        self.critic = TRPOCritic(self.agent_params)

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, ob_no, ac_no, re_n, next_ob_no, terminal_n):

        """
            Training a TRPO agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        # calculate advantages and target returs for value_function that correspond to each (s_t, a_t) point
        advantages, targets = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)

        loss = OrderedDict()

        #print(self.actor.parameters())
        loss['critic_loss'] = self.critic.update(ob_no, targets)
        loss['agent_loss']  = self.actor.update(ob_no, ac_no, advantages)


        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):

        """
            Computes advantages (both gae and standard) from the estimated Q values 
        """

        v_t = self.critic.forward_np(ob_no)
        v_tp1 = self.critic.forward_np(next_ob_no)
        

        if self.use_gae:
            last_gae = 0
            gaes = np.zeros(re_n.shape[0])

            for i in range(re_n.shape[0]-1, -1, -1):
                next_value = v_tp1[i]
                value      = v_t[i]
                delta    = re_n[i] + (self.gamma * next_value *(1-terminal_n[i])) - value
                last_gae = delta + self.gamma* self.lam * last_gae * (1-terminal_n[i])
                gaes[i]  = last_gae
            valuefn_targets = gaes + v_t
            advantages = gaes
        else:
            q_value = re_n + self.gamma * (v_tp1 * (1-terminal_n))
            valuefn_targets = q_value
            advantages = q_value - v_t

        # Normalize the resulting advantages
        if self.standardize_advantages:
            advantages = normalize(advantages, np.mean(advantages), np.std(advantages))

        return advantages, valuefn_targets

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def save(self, path):
        torch.save({
            "actor" : self.actor.state_dict(),
            "critic" : self.critic.state_dict(),
            "actor_optimizer": self.actor.optimizer.state_dict(),
            "critic_optimizer": self.critic.optimizer.state_dict()
            }, path)
        