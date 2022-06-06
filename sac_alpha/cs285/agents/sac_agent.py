import numpy as np
from collections import OrderedDict
import torch
from torch import optim
import math

from .base_agent import BaseAgent
from cs285.policies.MLP_policy import SAC_Alpha_Policy
from cs285.infrastructure.sac_replay_buffer import ReplayBuffer, Transition
from cs285.critics.sac_critic import SACCritic

from cs285.infrastructure.utils import *
from cs285.infrastructure import pytorch_util as ptu

class SACAgent(BaseAgent):
    def __init__(self, env, train_batch_size, agent_params, restore_path):
        super(SACAgent, self).__init__()

        # init vars
        self.env = env
        self.train_batch_size = train_batch_size
        self.gamma = agent_params['gamma']

        action_low, action_high = self.env.action_space.low, self.env.action_space.high
        self.target_entropy = -np.prod(self.env.action_space.shape)

        
        self.learning_starts = agent_params['learning_starts']
        self.learning_freq   = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']
        self.explore_steps = agent_params['exploration_steps']
        self.polyak_tau = agent_params['polyak_tau']

        # actor/policy
        self.actor = SAC_Alpha_Policy(
            agent_params['ac_dim'],
            agent_params['ob_dim'],
            agent_params['n_layers'],
            agent_params['size'],
            action_low,
            action_high,            
            discrete=agent_params['discrete'],
            learning_rate=agent_params['learning_rate_policyfn'],
        )

        self.q1 = SACCritic(agent_params)
        self.q2 = SACCritic(agent_params)

        self.q1_target = SACCritic(agent_params)
        self.q2_target = SACCritic(agent_params)

        self.alpha = torch.exp(torch.zeros(1, device=ptu.device)).requires_grad_()
        self.alpha_lr = agent_params['learning_rate_alpha']
        self.alpha_optimizer = optim.Adam([self.alpha], self.alpha_lr)

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)
        self.t = 0

        if restore_path:
            checkpoint = torch.load(restore_path)
            self.actor.load_state_dict(checkpoint['sac_actor'])
            self.q1.load_state_dict(checkpoint['q1_critic'])
            self.q2.load_state_dict(checkpoint['q2_critic'])
            self.q1_target.load_state_dict(checkpoint['q1_target'])
            self.q2_target.load_state_dict(checkpoint['q2_target'])
            self.alpha.data = checkpoint['alpha'].data

            self.actor.optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.q1.optimizer.load_state_dict(checkpoint['q1_optimizer'])
            self.q2.optimizer.load_state_dict(checkpoint['q2_optimizer'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])

            self.t = int(restore_path.split('.')[-2].split('_')[-1])+ 1


        self.last_obs = self.env.reset()

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """      
        perform_random_action = self.t < self.explore_steps
        if perform_random_action:
            # take random action 
            action = self.env.action_space.sample()
        else:
            action = self.actor.get_action(self.last_obs)
        
        next_obs, reward, done, info = self.env.step(action)
        transition = Transition(self.last_obs, action, reward, next_obs, done)

        self.replay_buffer.add_transition(transition)
        # if taking this step resulted in done, reset the env (and the latest observation)

        self.last_obs = next_obs
        if done:
            self.last_obs = self.env.reset()

    def train(self, ob_no, ac_no, re_n, next_ob_no, terminal_n):

        """ 
            Updating SAC actor using the given observations/actions and the seen rewards.
        """
        loss = OrderedDict()

        if (len(self.replay_buffer) > self.train_batch_size and self.t % self.learning_freq == 0):

            ob_no = ptu.from_numpy(ob_no)
            ac_no = ptu.from_numpy(ac_no)
            re_n =  ptu.from_numpy(re_n)
            next_ob_no = ptu.from_numpy(next_ob_no)
            terminal_n = ptu.from_numpy(terminal_n)

            with torch.no_grad():
                next_actions, next_logprobs = self.actor.actions_and_logprobs(next_ob_no)
                q1_targets = self.q1_target(next_ob_no, next_actions)
                q2_targets = self.q2_target(next_ob_no, next_actions)

                q_targets = torch.min(q1_targets, q2_targets) - self.alpha * next_logprobs
                targets = re_n + self.gamma * (1-terminal_n) * q_targets

            loss['q_1_loss'] = self.q1.update(ob_no, ac_no, targets)
            loss['q_2_loss'] = self.q2.update(ob_no, ac_no, targets)

            actions, log_probs = self.actor.actions_and_logprobs(ob_no)

            q_1 = self.q1(ob_no, actions)
            q_2 = self.q2(ob_no, actions)

            q_values = torch.min(q_1, q_2)

            policy_loss = (self.alpha * log_probs - q_values).mean()

            self.actor.optimizer.zero_grad()
            policy_loss.backward()
            self.actor.optimizer.step()

            alpha_loss =  (-self.alpha * ((log_probs.detach() + self.target_entropy))).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            loss['policy_loss'] = policy_loss.item()
            loss['alpha_loss'] = alpha_loss.item()

            if self.t % self.target_update_freq == 0:
                q1_params = ptu.flatten_params(self.q1.parameters())
                q1_target_params = ptu.flatten_params(self.q1_target.parameters())

                ptu.assign_params_to(self.polyak_tau * q1_target_params + (1-self.polyak_tau) * q1_params, 
                                                                            self.q1_target.parameters())

                q2_params = ptu.flatten_params(self.q2.parameters())
                q2_target_params = ptu.flatten_params(self.q2_target.parameters())

                ptu.assign_params_to(self.polyak_tau * q2_target_params + (1-self.polyak_tau) * q2_params, 
                                                                             self.q2_target.parameters())

        self.t += 1
            
        return loss

    #####################################################
    #####################################################

    def sample(self, batch_size):
        if len(self.replay_buffer) > self.train_batch_size:
            return self.replay_buffer.sample_random_data(batch_size)
        else:
            return [], [], [], [], []

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def save(self, path):
        torch.save({
            "sac_actor" : self.actor.state_dict(),
            "actor_optimizer": self.actor.optimizer.state_dict(),
            "q1_critic" : self.q1.state_dict(),
            "q2_critic" : self.q2.state_dict(),
            "q1_target" : self.q1_target.state_dict(),
            "q2_target" : self.q2_target.state_dict(),
            "q1_optimizer": self.q1.optimizer.state_dict(),
            "q2_optimizer": self.q2.optimizer.state_dict(),
            "alpha": self.alpha,
            "alpha_optimizer": self.alpha_optimizer.state_dict()
            }, path)
        