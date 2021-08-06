import numpy as np
from collections import OrderedDict
import torch
import math

from .base_agent import BaseAgent
from cs285.policies.MLP_policy import PPOPolicy
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.critics.ppo_critic import PPOCritic

from cs285.infrastructure.utils import *

class PPOAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PPOAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.use_gae = self.agent_params['use_gae']
        self.lam = self.agent_params['gae_lam']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.ppo_epochs = self.agent_params['ppo_epochs']
        self.ppo_min_bacth_size = self.agent_params['ppo_min_batch_size']

        # actor/policy
        self.actor = PPOPolicy(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['clip_eps'],
            self.agent_params['ent_coeff'],
            self.agent_params['max_grad_norm'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate_policyfn'],
        )

        self.critic = PPOCritic(self.agent_params)

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, ob_no, ac_no, re_n, next_ob_no, terminal_n, logprobs):

        """
            Training a PPO agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        # calculate advantages and target returs for value_function that correspond to each (s_t, a_t) point
        advantages, targets = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)

        # step 1: use all datapoints (s_t, a_t, q_t, adv_t) to update the PPO actor/policy
        ## HINT: `train_log` should be returned by your actor update method
        loss = OrderedDict()

        #print(self.actor.parameters())

        if self.ppo_min_bacth_size:

            n_batches = math.ceil(terminal_n.shape[0]/self.ppo_min_bacth_size)
            inds = np.arange(terminal_n.shape[0])

            for _ in range(self.ppo_epochs):
                np.random.shuffle(inds)

                for i in range(n_batches):

                    rand_indices = inds[slice(i*self.ppo_min_bacth_size, 
                                                     min(inds.shape[0], ((i+1) * self.ppo_min_bacth_size)))]
                    mb_ob_no = ob_no[rand_indices]
                    mb_ac_no = ac_no[rand_indices]
                    mb_adv = advantages[rand_indices]
                    mb_targets = targets[rand_indices]
                    mb_logprobs = logprobs[rand_indices]

                    loss['critic_loss'] = self.critic.update(mb_ob_no, mb_targets)
                    loss['agent_loss']  = self.actor.update(mb_ob_no, mb_ac_no, mb_adv, mb_logprobs)

        else:

            for _ in range(self.ppo_epochs):
                loss['critic_loss'] = self.critic.update(ob_no, targets)
                loss['agent_loss']  = self.actor.update(ob_no, ac_no, advantages, logprobs)

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
        