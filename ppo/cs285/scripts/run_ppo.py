import os
import time

from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.ppo_agent import PPOAgent

class PPO_Trainer(object):

    def __init__(self, params):

        #####################
        ## SET AGENT PARAMS
        #####################

        computation_graph_args = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            }

        estimate_advantage_args = {
            'gamma': params['discount'],
            'use_gae':params['use_gae'],
            'gae_lam':params['gae_lam'],
            'standardize_advantages': not(params['dont_standardize_advantages']),
        }

        ppo_update_args = {
            'clip_eps' : params['clip_epsilon'],
            'ent_coeff' : params['ent_coeff'],
            'max_grad_norm' : params['max_grad_norm'],
            'ppo_epochs': params['ppo_epochs'],
            'ppo_min_batch_size':params['ppo_min_batch_size']
        }

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
            'l2_reg': params['l2_reg'],
            'learning_rate_valuefn': params['learning_rate_valuefn'],
            'learning_rate_policyfn': params['learning_rate_policyfn'],
            'num_target_updates': params['num_target_updates'],
        }

        agent_params = {**computation_graph_args, **estimate_advantage_args, **ppo_update_args, **train_args}

        self.params = params
        self.params['agent_class'] = PPOAgent
        self.params['agent_params'] = agent_params
        self.params['batch_size_initial'] = self.params['batch_size']

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            self.params['n_iter'],
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--n_iter', '-n', type=int, default=200)

    parser.add_argument('--use_gae', action='store_true')
    parser.add_argument('--gae_lam', type=float, default=0.95)
    parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true')
    parser.add_argument('--batch_size', '-b', type=int, default=2048) #steps collected per train iteration
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=5000) #steps collected per eval iteration
    parser.add_argument('--ppo_min_batch_size', type=int, default=64)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--num_target_updates', '-ntu', type=int, default=1)
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--learning_rate_valuefn', '-lr_v', type=float, default=3e-4)
    parser.add_argument('--learning_rate_policyfn', '-lr_p', type=float, default=3e-4)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)

    # PPO update params
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='clip epsilon for ppo objective')
    parser.add_argument('--ent_coeff', type=float, default=0, help='coefficient for entropy bonus')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, 
                                           help='used for gradient clipping in policy update')
    parser.add_argument('--ppo_epochs', type=int, default=10, help='number of ppo updates in one \
                                                                               training iteration')
    parser.add_argument('--l2_reg', type=float, default=1e-3,
                                           help='l2 regression coefficient for critic network')

    parser.add_argument('--ep_len', type=int) #students shouldn't change this away from env's default
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--scalar_log_freq', type=int, default=1)

    parser.add_argument('--save_params', action='store_true')

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    ## ensure compatibility with hw1 code
    params['train_batch_size'] = params['batch_size']

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.env_name + '_' + args.exp_name + '_' + time.strftime("%d-%m-%Y-%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    ###################
    ### RUN TRAINING
    ###################

    trainer = PPO_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
