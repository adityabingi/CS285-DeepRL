import os
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from glob import glob
from collections import OrderedDict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SEEDS = sorted(random.sample(range(1, 1000), 5))

def run_exp1_1():

    cmd = "python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
            --exp_name {exp} --seed {seed} --eval_batch_size 1000"

    # reward-to-go rtg; don't standardize advantages dsa;
    cmdline_args = [' -dsa', ' -dsa -rtg', ' -rtg']
    exp_names = ['q1_sb_no_rtg_dsa', 'q1_sb_rtg_dsa', 'q1_sb_rtg_na']

    
    for i, exp_name in enumerate(exp_names):
        for seed in SEEDS:
            new_cmd = cmd + cmdline_args[i]
            new_cmd = new_cmd.format(exp = exp_name +'_seed' +str(seed), seed=seed)
            os.system(new_cmd)

def run_exp1_2():

    cmd = "python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 \
           --exp_name {exp} --seed {seed} --eval_batch_size 1000"

    # reward-to-go rtg; don't standardize advantages dsa;
    cmdline_args = [' -dsa', ' -dsa -rtg', ' -rtg']
    exp_names = ['q1_lb_no_rtg_dsa', 'q1_lb_rtg_dsa', 'q1_lb_rtg_na']

    for i, exp_name in enumerate(exp_names):
        for seed in SEEDS:
            new_cmd = cmd + cmdline_args[i]
            new_cmd = new_cmd.format(exp = exp_name +'_seed' +str(seed), seed=seed)
            os.system(new_cmd)


# find best batch-size b and lr_rate lr first by random-searching over batch_size and learning-rates
# pass found best batch_size and learning_rate as args to run this experiment
def run_exp2(batch_size, learning_rate):

    cmd = "python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 \
          --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b {b} -lr {lr} -rtg \
          --exp_name {exp} --seed {seed} --eval_batch_size 5000"

    for seed in SEEDS:
        new_cmd = cmd.format(exp = f'q2_b{batch_size}_r{learning_rate}'  +'_seed' +str(seed), 
                            b = batch_size, lr=learning_rate, seed=seed)
        os.system(new_cmd)

def run_exp3():

    cmd = "python cs285/scripts/run_hw2.py --env_name LunarLanderContinuous-v2 --ep_len 1000 \
           --discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 \
           --reward_to_go --nn_baseline --exp_name {exp} --seed {seed}\
           --eval_batch_size 5000"

    for seed in SEEDS:
        new_cmd = cmd.format(exp = 'q3_b40000_r0.005' +'_seed' +str(seed), seed=seed)
        os.system(new_cmd)

def run_exp4_1():

    cmd = "python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
           --discount 0.95 -n 100 -l 2 -s 32 -b {b} -lr {lr} -rtg --nn_baseline \
           --exp_name {exp} --eval_batch_size 1500"

    batch_sizes = [10000, 30000, 50000]
    lr_rates = [0.005, 0.01, 0.02]

    for b in batch_sizes:
        for lr_rate in lr_rates:
            new_cmd = cmd.format(b=b, lr=lr_rate, exp='q4_search_b'+ str(b) + '_lr' + str(lr_rate) +'_rtg_nnbaseline')
            os.system(new_cmd)

def run_exp4_2(batch_size, learning_rate):

    cmd = "python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 \
          --discount 0.95 -n 100 -l 2 -s 32 -b {b} -lr {lr} \
          --exp_name {exp} --seed {seed} --eval_batch_size 1500"

    cmdline_args = ['',' -rtg', ' --nn_baseline', ' -rtg --nn_baseline']
    exp_names = ['', '_rtg', '_nnbaseline', '_rtg_nnbaseline']

    for i, exp_name in enumerate(exp_names):
        for seed in SEEDS:
            new_cmd = cmd + cmdline_args[i]
            print(new_cmd)
            new_cmd = new_cmd.format(exp = f'q4_b{batch_size}_r{learning_rate}' + exp_name + '_seed' +str(seed), 
                                     b = batch_size, lr = learning_rate, seed=seed)
            os.system(new_cmd)

# Plot Arguments
exp1_1 = {

"exp_name": 'CartPole-v0_q1_sb',
"plot_title": 'CartPole-v0 pg agent trained with 1000 batch size',
"plot_labels": ['vanilla pg', 'rtg', 'rtg & std adv'],
"num_iter": 100,
"seeds": SEEDS

}

exp1_2 = {

"exp_name": 'CartPole-v0_q1_lb',
"plot_title": 'CartPole-v0 pg agent trained with 5000 batch size',
"plot_labels": ['vanila pg', 'rtg', 'rtg & std adv'],
"num_iter": 100,
"seeds": SEEDS

}

# Rando
exp2 ={

"exp_name": 'InvertedPendulum-v2_q2_b',
"plot_title": 'InvertedPendulum-v2 pg agent for single seed with batch-size {b} and learning-rate {lr}',
"plot_labels": ['rtg & std_adv'],
"num_iter": 100,
"seeds":[1]

}

exp3 ={

"exp_name": 'LunarLanderContinuous-v2_q3_b',
"plot_title": 'LunarLanderContinuous-v2 pg agent trained with rtg nn_baseline and standardized-advantages',
"plot_labels": ['rtg & nn_baseline'],
"num_iter": 100,
"seeds":SEEDS

} 

exp4_1 ={

"exp_name": 'HalfCheetah-v2_q4_search',
"plot_title": 'HalfCheetah-v2 pg agent trained with rtg nn_baseline and standardized-advantages for single seed',
"plot_labels": ['bs=10000-lr=0.005', 'bs=10000-lr=0.01', 'bs=10000-lr=0.02',
                'bs=30000-lr=0.005', 'bs=30000-lr=0.01', 'bs=30000-lr=0.02',
                'bs=50000-lr=0.005', 'bs=50000-lr=0.01', 'bs=50000-lr=0.02'],
"num_iter": 100,
"seeds": [1]

}

exp4_2 ={

"exp_name": 'HalfCheetah-v2_q4_b',
"plot_title": 'HalfCheetah-v2 pg agent trained with batch-size {b} and learning-rate {lr} & standardized-advantages',
"plot_labels": ['vanilla pg', 'rtg', 'nn_baseline', 'rtg & nn_baseline'],
"num_iter": 100,
"seeds": SEEDS

}


def make_plot(plot_args, cmd_line_args):

    """
    Utility function for plotting mean_rewards against number of training iterations

    """
    exp_name = plot_args['exp_name']
    plot_labels = plot_args['plot_labels']
    plot_title = plot_args['plot_title']

    if cmd_line_args.exp =='2' or cmd_line_args.exp == '4_2':
        plot_title = plot_title.format(b=cmd_line_args.batch_size, lr=cmd_line_args.learning_rate)

    n_iter = plot_args['num_iter']
    seeds  = plot_args['seeds']

    print(plot_labels)

    scalars = ["Eval_AverageReturn", "Eval_StdReturn"]
    logdir_names = list(glob('data/' + exp_name +'*/'))
    logdirs = sorted(logdir_names, key = lambda x: x.split("_")[-1])

    print(logdirs)

    sns.set(style='darkgrid')
    fig, ax = plt.subplots(figsize =(10,7))

    mean_rewards = np.zeros(shape=(len(plot_labels), n_iter))
    std_rewards  = np.zeros(shape=(len(plot_labels), n_iter))


    for i in range(0, len(plot_labels)):
        for j in range(len(seeds)):
            #logdir  = 'data/{}_{}_seed{}/'.format(env, log_name, seed)
            logfile = os.path.join(logdirs[i * len(seeds)+j], 'events*')
            logfile = glob(logfile)[0]
            loaded_scalars = load_tensorboard_logs(logfile, scalars[:2])
            for event in loaded_scalars[0]:
                mean_rewards[i][event[1]] += event[2]
            for event in loaded_scalars[1]:
                std_rewards[i][event[1]] += np.square(event[2])

        mean_rewards[i] = mean_rewards[i]/ len(seeds)
        std_rewards[i]  = np.sqrt(std_rewards[i]) / len(seeds)

        ax = plot_metrics(ax, list(range(1, n_iter+1)), mean_rewards[i], std_rewards[i],
                       label=plot_labels[i])


    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Mean_Reward')
    #ax.set_xticks(list(range(1, n_iter+1)))

    ax.legend(loc="best")
    ax.set_title(plot_title)

    fig.savefig('data/exp_{}.jpg'.format(exp_name))


def plot_metrics(ax, x, y, stds=None, label=None):

    x = np.array(x)
    y = np.array(y)

    # std fill_between

    ax.plot(x, y, label=label)
    if stds is not None:
        ax.fill_between(x, y-stds, y+stds, alpha=0.2)
    """if stds is not None:
        ax.errorbar(x, y, yerr=stds, label=label, c=c)"""

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


    loaded_scalars = []
    for scalar in scalars:
        loaded_scalars.append(event_acc.Scalars(scalar))

    return loaded_scalars


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp',
        type = str,
        default='1_1',
        choices=('1_1', '1_2' ,'2', '3', '4_1', '4_2')
    )
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--batch_size', type=int, default=None, help ='only for exp 2 and 4_2')
    parser.add_argument('--learning_rate', type=float, default=None, help ='only for exp2 and 4_2')
    args = parser.parse_args()

    if args.exp == '1_1':
        if args.run:
            run_exp1_1()
        if args.plot:
            make_plot(exp1_1, args)
    elif args.exp == '1_2':
        if args.run:
            run_exp1_2()
        if args.plot:
            make_plot(exp1_2, args)
    elif args.exp == '2':
        if args.run:
            run_exp2(args.batch_size, args.learning_rate)
        if args.plot:
            make_plot(exp2, args)
    elif args.exp == '3':
        if args.run:
            run_exp3()
        if args.plot:
            make_plot(exp3, args)
    elif args.exp == '4_1':
        if args.run:
            run_exp4_1()
        if args.plot:
            make_plot(exp4_1, args)
    else:
        if args.run:
            run_exp4_2(args.batch_size, args.learning_rate)
        if args.plot:
            make_plot(exp4_2, args)

if __name__ == '__main__':
    main()
