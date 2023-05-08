#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pickle
import os
import os.path
import sys

class LearningCurvePlot:
    '''
    A class for handling better a learning curve. This is because #curves = #repetitions for each combination.
    Also we apply smoothing to every learning curve. 
    This code is taken from the previous assignment after getting permission.  
    '''
    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots(figsize=(8, 5), layout="constrained")
        self.ax.set_xlabel('Episode', fontsize=13)
        self.ax.set_ylabel('Reward', fontsize=13)
        if title is not None:
            self.ax.set_title(title, fontsize=15)
        
    def add_curve(self, y, label=None):
        if label is not None:
            self.ax.plot(y, label=label)
        else:
            self.ax.plot(y)

    def add_confidence_interval(self, y, interval, label=None):
        if label is not None:
            self.ax.fill_between(x=np.arange(len(y)), y1=y-interval, y2=y+interval, alpha=0.2, label=label)
        else:
            self.ax.fill_between(x=np.arange(len(y)), y1=y-interval, y2=y+interval, alpha=0.2, label=label)
    
    def set_ylim(self, lower, upper):
        self.ax.set_ylim([lower, upper])

    def add_hline(self, height, label=None):
        if label is not None:
            self.ax.axhline(height, ls='--', c='k', label=label)
        else:
            self.ax.axhline(height, ls='--', c='k')

    def save(self, name='test.png', legend_title=None):
        ''' name: string for filename of saved figure '''
        self.ax.legend(bbox_to_anchor=(1, 1), title=legend_title, alignment='left')
        self.fig.savefig(name, dpi=300)

def ckeckCMD():
    '''
    Ckeck the command line from the terminal for validity and print the corresponding message (if needed).
    '''
    acceptedTokens = ['actor_critic.py','--bootstr','--basesub']
    if len(sys.argv)<1 or len(sys.argv)>3:
        return 'error_length'
    for term in sys.argv:
        if term not in acceptedTokens:
            return 'error_term'
        
    if len(sys.argv) == 1:
        return 'AC'
    elif '--bootstr' in sys.argv and '--basesub' not in sys.argv:
        return 'BT'
    elif '--basesub' in sys.argv and '--bootstr' not in sys.argv:
        return 'BS'
    elif '--bootstr' in sys.argv and '--basesub' in sys.argv:
        return 'BT_BS'

def printNotAcceptedCMD(error_msg):
    '''
    Print error message in case checkCMD() function returns error code.
    param error_msg:    the error message from the given error command line 
    '''
    if error_msg == 'error_length':
        print('The command was not accepted due to the following reason: too many/few arguments')
    if error_msg == 'error_term':
        print('The command was not accepted due to the following reason: wrong argument(s) given')
    print('Please follow one of the examples listed below:')
    print('$ python actor_critic.py')
    print('$ python actor_critic.py --bootstr')
    print('$ python actor_critic.py --basesub')
    print('$ python actor_critic.py --bootstr --basesub')

def get_height(rows: int = 7, speed: float = 1):
    '''
    Returns value for the maximal reward possible in the environment, depending on the number of rows
    and speed of new balls being dropped.
    param rows: number of rows in the environment
    param speed: speed setting of the environment
    '''
    return int((250-rows+1)/int(rows/speed)+1)

def smooth(y, window, poly=1):
    '''
    Smooth the learning curve with savgol_filter.
    param y:        vector to be smoothed 
    param window:   size of the smoothing window '''
    return savgol_filter(y, window, poly)

def make_central_directory(target):
    '''
    Make the central directory for storing all the results.
    param target:   name of the central directory
    '''
    file_dir = os.path.dirname(__file__)
    central_path = file_dir + target
    try:
        os.mkdir(central_path)
    except OSError as error:
        print(error)
    return central_path

def saveResults(exp, comb, reward_results):
    '''
    save results
    exp: name of the experiment
    comb: combination of variables tested
    reward_results: output after n_reps repetitions
    '''
    central_path = make_central_directory('/{}'.format(exp))
    saved_dict = dict()
    saved_dict[comb] = reward_results
    name = central_path + '/combination_' + str(comb) + '.pkl'
    a_file = open(name, 'wb')
    pickle.dump(saved_dict,a_file,pickle.HIGHEST_PROTOCOL)
    a_file.close()

def plot_smooth(exp, smooth_win):
    file_dir = os.path.dirname(__file__)

    if exp == 'part1_reinforce':
        plot_title = r'REINFORCE: exploring learning rate and $\eta$'
        legend_title = r'learning rate, $\eta$'
        central_path = file_dir + chr(92) + exp
    elif 'tuning' in exp:
        if exp == 'AC_baseline tuning':
            plot_title = 'Actor Critic with Baseline Subtraction'
        elif exp == 'AC_both tuning':
            plot_title = 'Actor Critic with Baseline Subtraction and Bootstrapping'
        elif exp == 'AC_boot tuning':
            plot_title = 'Actor Critic with Bootstrapping'
        legend_title = r'learning rate, $\eta$'
        central_path = file_dir + r'\RL3\learning_rate_and_eta' + chr(92) + exp + ' new'
    elif 'n_boot' in exp:  # bootstrapping experiments
        legend_title = 'Number of samples'
        central_path = file_dir + r'\RL3' + chr(92) + exp
        if 'AC_baseline' in exp:
            plot_title = 'Actor Critic with Baseline Subtraction'
        elif 'AC_boot' in exp:
            plot_title = 'Actor Critic with Bootstrapping'
        elif 'AC_both' in exp:
            plot_title = 'Actor Critic with Baseline Subtraction and Bootstrapping'
    else:  # environment experiments
        central_path = file_dir + r'\RL3' + chr(92) + 'environment' + chr(92) + exp
        if exp == 'vector':
            plot_title = "Environment: Varying the Observation Type"
            legend_title = 'Observation Type'
        elif exp == 'speed':
            plot_title = "Environment: Varying the Speed"
            legend_title = 'speed'
        elif exp == 'sizes':
            plot_title = "Environment: Varying the Environment Size"
            legend_title = 'Rows, Columns'
        elif exp == 'combine':
            central_path = file_dir + r'\RL3' + chr(92) + 'environment' + chr(92) + exp + chr(92) + '2k'
            plot_title = "Environment: Exploring a Combination of Changes"
            legend_title = None
        elif exp == 'other':
            plot_title = 'Environment: Exploring a change in Width and Speed'
            legend_title = None

    plot = LearningCurvePlot(title=plot_title)

    for f in os.listdir(central_path):
        if '.pkl' in f and '0.0001' not in f:
            target_f = central_path + chr(92) + f
            a_file = open(target_f, 'rb')
            x_dict = pickle.load(a_file)
            reward_results = x_dict[list(x_dict.keys())[0]]
            exp_settings = list(x_dict.keys())[0]
            a_file.close()

            learning_curve = np.mean(reward_results, axis=0)  # average over repetitions
            if smooth_win != False:
                learning_curve = smooth(learning_curve, smooth_win)  # additional smoothing
            learning_curve_std = np.std(reward_results, axis=0)  # calculate standard deviation for confidence interval

            if exp == 'part1_reinforce' or 'tuning' in exp:
                plot.add_curve(learning_curve, label=f'{exp_settings[0]}, {exp_settings[1]}')
            elif 'n_boot' in exp:
                plot.add_curve(learning_curve, label=f'{exp_settings[-1]}')
            elif exp == 'vector':
                plot.add_curve(learning_curve, label=f'{exp_settings[5]}')
            elif exp == 'speed':
                plot.add_curve(learning_curve, label=f'{exp_settings[-1]}')
            elif exp == 'sizes':
                plot.add_curve(learning_curve, label=f'{exp_settings[3]}, {exp_settings[4]}')
            else:
                plot.add_curve(learning_curve)  # , label=exp_settings)

            plot.add_confidence_interval(learning_curve, learning_curve_std)

    if exp != 'other' and exp != 'combine':
        plot.add_hline(height=get_height(rows=7, speed=1))
    if exp == 'other':
        plot.add_hline(height=get_height(rows=7, speed=2))
    if exp == 'sizes':
        plot.add_hline(height=get_height(rows=10, speed=1))
        plot.add_hline(height=get_height(rows=3, speed=1))
    if exp == 'speed':
        plot.add_hline(height=get_height(rows=7, speed=2))
        plot.add_hline(height=get_height(rows=7, speed=0.5))
    if exp == 'combine':
         plot.add_hline(height=get_height(rows=7, speed=0.5))

    if 'n_boot' not in exp:
        plot.save('{}_win{}.png'.format(exp, smooth_win), legend_title=legend_title)
        plot.save('{}_win{}.svg'.format(exp, smooth_win), legend_title=legend_title)
    else:
        plot.save('n_boot{}_win{}.png'.format(exp[7:], smooth_win), legend_title=legend_title)
        plot.save('n_boot{}_win{}.svg'.format(exp[7:], smooth_win), legend_title=legend_title)




if __name__ == '__main__':

    plot = LearningCurvePlot(title=r'Actor Critic with varying $\eta$')

    eta_0_folder = os.path.dirname(__file__) + r'\eta_0' + chr(92)
    for i, f in enumerate(os.listdir(eta_0_folder)):
        print(f)
        a_file = open(eta_0_folder+f, 'rb')
        x_dict = pickle.load(a_file)
        reward_results = x_dict[list(x_dict.keys())[0]]
        exp_settings = list(x_dict.keys())[0]
        a_file.close()

        learning_curve = np.mean(reward_results, axis=0)  # average over repetitions
        learning_curve_std = np.std(reward_results, axis=0)


        plot.add_curve(learning_curve, label=exp_settings[1])
        plot.add_confidence_interval(learning_curve, learning_curve_std)
        plot.add_hline(height=get_height(rows=7, speed=1))

    plot.save('AC_eta_0', legend_title=r'$\eta$')
