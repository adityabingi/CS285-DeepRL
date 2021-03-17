import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from glob import glob
from collections import OrderedDict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# script for running hw1 experiments

EXPERTS = ['Ant-v2','Humanoid-v2']

def run_bc(env):

    """ Behaviour cloning experiments according to Sections 1_2 and 1_3 of cs285_hw1.pdf """

    hparams = OrderedDict()
    hparams['train_steps_per_iter'] = list(range(1000, 5000+1, 1000))
    hparams['lr_rate']    = list(1e1 ** np.array(range(-4, -1, 1)))

    cmd = "python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/{policy}.pkl \
           --env_name {env} --exp_name {exp} --n_iter 1 \
           --expert_data cs285/expert_data/expert_data_{data}.pkl\
           --num_agent_train_steps_per_iter {train_steps} --seed {seed} --eval_batch_size 5000\
           --video_log_freq -1"

    seeds = range(5)

    for seed in seeds:
        for train_steps in hparams['train_steps_per_iter']:
            new_cmd = cmd.format(env = env, exp ='bc_'+ 'train_steps_per_iter' + str(train_steps)+'_seed' +str(seed), seed=seed, 
                                 policy= env.split("-")[0], data=env, train_steps = train_steps)
            os.system(new_cmd)

    visualize_bc(env, hparams['train_steps_per_iter'], 'train_steps_per_iter', seeds)

    cmd = "python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/{policy}.pkl \
           --env_name {env} --exp_name {exp} --n_iter 1 \
           --expert_data cs285/expert_data/expert_data_{data}.pkl\
           --learning_rate {lr_rate} --seed {seed} --eval_batch_size 5000\
           --video_log_freq -1"


    for seed in seeds:
        for lr_rate in hparams['lr_rate']:
            new_cmd = cmd.format(env = env, exp = 'bc_' +'lr_rate' + str(lr_rate) + '_seed'+str(seed), seed=seed,
                                 policy= env.split("-")[0], data=env, lr_rate =lr_rate)
            os.system(new_cmd)

    visualize_bc(env, hparams['lr_rate'], 'lr_rate', seeds)

def visualize_bc(env, hparam_values, hparam_key, seeds):

    scalars = ["Eval_MeanReturn", "Eval_StdReturn", "Initial_DataCollection_MeanReturn"]

    sns.set(style='darkgrid')
    fig, ax = plt.subplots(figsize = (10,7))

    mean_rewards = np.zeros(shape=len(hparam_values))
    std_rewards  = np.zeros(shape=len(hparam_values))
    expert_mean_rewards = np.zeros(shape=len(hparam_values))
    
    for seed in seeds:  
        for i, hparam in enumerate(hparam_values):
            logdir  = 'data/q1_bc_{}{}_seed{}_{}/'.format(hparam_key, str(hparam), seed, env)
            logfile = os.path.join(logdir, 'events*')
            logfile = glob(logfile)[0]
            loaded_scalars = load_tensorboard_logs(logfile, scalars)
            mean_rewards[i] += loaded_scalars[0][0][2]
            std_rewards[i]  += np.square(loaded_scalars[1][0][2])
            expert_mean_rewards[i] += loaded_scalars[2][0][2]
            
    mean_rewards = mean_rewards / len(seeds)
    std_rewards  = np.sqrt(std_rewards) / len(seeds)
    expert_mean_rewards = expert_mean_rewards/ len(seeds)

    ax = plot_metrics(ax, hparam_values, mean_rewards, std_rewards,
                    label='bc_agent_{}'.format(env), c='r')
    ax = plot_metrics(ax, hparam_values, expert_mean_rewards,
                    label='expert_agent_{}'.format(env), c='g')

    ax.set_xlabel(hparam_key)
    ax.set_ylabel('Mean_Reward')
    ax.set_xticks(hparam_values)

    ax.legend(loc='best')
    ax.set_title('Agents with mean&std rewards averaged over 5 rollouts and 5 random seeds')

    fig.savefig('data/bc_{}_{}.jpg'.format(hparam_key, env))

def run_dagger(env):

    """ Dagger experiments according to Section 2 of cs285_hw1.pdf """

    cmd = "python cs285/scripts/run_hw1.py --expert_policy_file cs285/policies/experts/{policy}.pkl \
          --env_name {env} --exp_name {exp} --n_iter {n_iter} \
          --do_dagger --expert_data cs285/expert_data/expert_data_{data}.pkl \
          --num_agent_train_steps_per_iter 1000 --eval_batch_size 5000 --seed {seed}\
          --video_log_freq -1"

    n_iter = 10
    seeds = range(5)
    for seed in seeds:
        new_cmd = cmd.format(env = env, exp ='dagger_'+ 'seed{}'.format(seed), n_iter=n_iter, 
                             policy= env.split("-")[0], data=env, seed = seed)
        os.system(new_cmd)

    visualize_dagger(env, n_iter, seeds)

def visualize_dagger(env, n_iter, seeds):

    scalars = ["Eval_MeanReturn", "Eval_StdReturn", "Initial_DataCollection_MeanReturn"]

    sns.set(style='darkgrid')
    fig, ax = plt.subplots(figsize =(10,7))

    mean_rewards = np.zeros(shape=n_iter)
    std_rewards  = np.zeros(shape=n_iter)

    expert_mean_rewards = np.zeros(shape=n_iter)
    bc_mean_rewards = np.zeros(shape=n_iter)
    bc_std_rewards = np.zeros(shape=n_iter)

    for seed in seeds:

        # dagger agent logs
        logdir  = 'data/q2_dagger_seed{}_{}/'.format(seed, env)
        logfile = os.path.join(logdir, 'events*')
        logfile = glob(logfile)[0]
        loaded_scalars = load_tensorboard_logs(logfile, scalars[:2])
        for event in loaded_scalars[0]:
            mean_rewards[event[1]] += event[2]
        for event in loaded_scalars[1]:
            std_rewards[event[1]] += np.square(event[2])

        # behaviour-cloning agent logs
        logdir = 'data/q1_bc_train_steps_per_iter1000_seed{}_{}'.format(seed, env)
        logfile = os.path.join(logdir, 'events*')
        logfile = glob(logfile)[0]
        loaded_scalars = load_tensorboard_logs(logfile, scalars)
        bc_mean_rewards[0] += loaded_scalars[0][0][2]
        bc_std_rewards[0] += np.square(loaded_scalars[1][0][2])
        expert_mean_rewards[0] += loaded_scalars[2][0][2]
        

    bc_mean_rewards.fill(bc_mean_rewards[0])
    bc_std_rewards.fill(bc_std_rewards[0])
    expert_mean_rewards.fill(expert_mean_rewards[0])


    mean_rewards = mean_rewards / len(seeds)
    std_rewards  = np.sqrt(std_rewards) / len(seeds)
    bc_mean_rewards = bc_mean_rewards/ len(seeds)
    bc_std_rewards = np.sqrt(bc_std_rewards) / len(seeds)
    expert_mean_rewards = expert_mean_rewards/ len(seeds)
    

    ax = plot_metrics(ax, list(range(1, n_iter+1)), mean_rewards, std_rewards,
                       label='dagger_agent_{}'.format(env), c = 'r')
    ax = plot_metrics(ax, list(range(1, n_iter+1)), bc_mean_rewards, bc_std_rewards,
                       label='bc_agent_{}'.format(env), c='b')
    ax = plot_metrics(ax, list(range(1, n_iter+1)), expert_mean_rewards, 
                       label='expert_agent_{}'.format(env), c='g')

    ax.set_xlabel('Number of Dagger Iterations')
    ax.set_ylabel('Mean_Reward')
    ax.set_xticks(list(range(1, n_iter+1)))

    ax.legend(loc="best")
    ax.set_title('Agents with mean&std rewards averaged over 5 rollouts and 5 random seeds')

    fig.savefig('data/dagger_{}.jpg'.format(env))


def plot_metrics(ax, x, y, stds=None, label=None, c='y'):

    x = np.array(x)
    y = np.array(y)

    if stds is not None:
        ax.errorbar(x, y, yerr=stds, label=label, c=c)
    else:
        ax.plot(x, y, label=label, c=c)
    
    # std fill_between
    """if stds is not None:
        ax.fill_between(x, y-stds, y+stds, alpha=0.2, facecolor=c)"""

    return ax

def load_tensorboard_logs(path, scalars):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 500,
        'images': 0,
        'scalars': 10000,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    #print(event_acc.Tags())

    loaded_scalars = []
    for scalar in scalars:
        loaded_scalars.append(event_acc.Scalars(scalar))

    return loaded_scalars


def main():

    for expert in EXPERTS:
        run_bc(expert)
        run_dagger(expert)

if __name__ == '__main__':
    main()