import os
import time

from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.td3_agent import TD3Agent

class TD3_Trainer(object):

    def __init__(self, params):

        #####################
        ## SET AGENT PARAMS
        #####################

        computation_graph_args = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'num_target_updates': params['num_target_updates'],
            }

        td3_update_args = {
            'gamma': params['discount'],
            'polyak_tau' : params['polyak_tau'],
            'learning_rate_valuefn': params['learning_rate_valuefn'],
            'learning_rate_policyfn': params['learning_rate_policyfn'],
            'policy_noise': params['policy_noise'],
            'noise_clip': params['noise_clip'],
            'expl_noise': params['expl_noise'],
        }

        train_args = {
            'exploration_steps': params['exploration_steps'],
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
            'l2_reg': params['l2_reg'],
            'learning_starts': params['learning_starts'],
            'learning_freq': params['learning_freq'],
            'target_update_freq': params['target_update_freq'],
        }

        agent_params = {**computation_graph_args,  **td3_update_args, **train_args}

        self.params = params
        self.params['agent_class'] = TD3Agent
        self.params['agent_params'] = agent_params
        self.params['batch_size_initial'] = self.params['batch_size']

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            self.params['total_timesteps'],
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='todo')
    parser.add_argument('--total_timesteps', '-n', type=int, default=int(2e8))

    parser.add_argument('--batch_size', '-b', type=int, default=256) #steps collected per train iteration
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=5000) #steps collected per eval iteration

    parser.add_argument('--learning_rate_valuefn', '-lr_v', type=float, default=3e-4)
    parser.add_argument('--learning_rate_policyfn', '-lr_p', type=float, default=3e-4)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=256)

    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--polyak_tau', type=float, default=0.995)
    parser.add_argument('--policy_noise', type=float, default=0.2)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--expl_noise', type=float, default=0.1)
    parser.add_argument('--l2_reg', type=float, default=1e-3,
                                           help='l2 regression coefficient for critic network')


    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--num_target_updates', '-ntu', type=int, default=1)
    parser.add_argument('--exploration_steps', type=int, default=10000)
    parser.add_argument('--learning_starts', type=int, default=1000)
    parser.add_argument('--learning_freq', type=int, default=1)
    parser.add_argument('--target_update_freq', type=int, default=1)

    parser.add_argument('--ep_len', type=int) #students shouldn't change this away from env's default
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--video_log_freq', type=int, default =- 1)
    parser.add_argument('--scalar_log_freq', type=int, default = 100)
    parser.add_argument('--checkpoint_freq', type=int, default=1000)

    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--restore_path', type=str)

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

    trainer = TD3_Trainer(params)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
