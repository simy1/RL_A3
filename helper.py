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

class LearningCurvePlot:

    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward')      
        if title is not None:
            self.ax.set_title(title)
        
    def add_curve(self, y, label=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
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

    def add_hline(self,height,label):
        self.ax.axhline(height,ls='--',c='k',label=label)

    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.legend()
        self.fig.savefig(name,dpi=300)

def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)

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
    if exp == 'reinforce':
        plot = LearningCurvePlot(title=r'REINFORCE: exploring learning rate and $\eta$')
    elif exp == 'actorcritic':
        plot = LearningCurvePlot(title='Exploring a combination of changes')

    file_dir = os.path.dirname(__file__)
    central_path = file_dir +'/'+ exp
    for f in os.listdir(central_path):
        target_f = central_path +'/'+ f
        a_file = open(target_f, 'rb')
        x_dict = pickle.load(a_file)
        reward_results = x_dict[list(x_dict.keys())[0]]
        a_file.close()

        learning_curve = np.mean(reward_results, axis=0)  # average over repetitions
        if smooth_win != False:
            learning_curve = smooth(learning_curve, smooth_win)  # additional smoothing
        learning_curve_std = np.std(reward_results, axis=0)
        plot.add_curve(learning_curve)
        plot.add_confidence_interval(learning_curve, learning_curve_std)
    
    plot.save('{}_win{}.png'.format(exp, smooth_win))
