#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

from tqdm import tqdm
import numpy as np
import time
import itertools
from torch import optim

from catch import Catch
from reinforce_const import Model, runReinforceAlgo
from helper import LearningCurvePlot, smooth, saveResults, plot_smooth
from actor_critic import *


def average_over_repetitions(exp, settings, comb, smoothing_window=5):

    reward_results = []  # Result array
    now = time.time()
    
    for _ in range(settings['n_reps']):  # Loop over repetitions
        if exp == 'part1_reinforce':
            env = Catch(rows=settings['rows'], columns=settings['columns'], speed=settings['speed'], max_steps=settings['max_steps'],
                    max_misses=settings['max_misses'], observation_type=settings['observation_type'], seed=settings['seed'])
            s = env.reset()

            model = Model(settings['observation_type'], settings['rows'], settings['columns'])
            optimizer = optim.Adam(model.neuralnetwork.parameters(), lr=comb[0])

            rewards_per_episode = runReinforceAlgo(env=env,
                                                   model=model, 
                                                   optimizer=optimizer,
                                                   gamma=settings['gamma'],
                                                   iterations=settings['iterations'], 
                                                   eta=comb[1],
                                                   print_details=settings['print_details'])

        elif exp == 'part1_actorcritic':
            env = Catch(rows=settings['rows'], columns=settings['columns'], speed=settings['speed'], max_steps=settings['max_steps'],
                    max_misses=settings['max_misses'], observation_type=settings['observation_type'], seed=settings['seed'])
            state = env.reset()
            n_input = len(state.flatten())
            state = torch.from_numpy(state.flatten()).float().unsqueeze(0)  # convert to tensor
            n_actions = env.action_space.n
            actor = Actor(n_input=n_input, n_actions=n_actions)
            critic = Critic(n_input=n_input, n_actions=n_actions)
            actor_optimizer = optim.Adam(actor.parameters(), lr=comb[0])
            critic_optimizer = optim.Adam(critic.parameters(), lr=comb[0])

            # perform the algorithm
            rewards_per_episode = run_episodes(env=env, actor=actor, critic=critic, actor_optimizer=actor_optimizer,
                                               critic_optimizer=critic_optimizer, bootstrap=settings['bootstrap'],
                                               baseline_subtraction=settings['baseline_subtraction'],
                                               n_episodes=settings['iterations'], n_boot=comb[2],
                                               gamma=settings['gamma'], entropy_reg_strength=comb[1])

        reward_results.append(rewards_per_episode)


    # save the results for this combination(comb) and for all repetitions(settings['n_reps'])
    saveResults(exp, comb, reward_results)
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))    
    learning_curve = np.mean(reward_results, axis=0)  # average over repetitions
    learning_curve = smooth(learning_curve, smoothing_window)  # additional smoothing
    learning_curve_std = np.std(reward_results, axis=0)
    return learning_curve, learning_curve_std


def experiment(exp):
    if exp == 'part1_reinforce':
        # settings (fixed/constant during experiments)
        settings = dict()
        settings['rows'] = 7
        settings['columns'] = 7
        settings['max_steps'] = 250
        settings['max_misses'] = 10
        settings['observation_type'] = 'pixel'
        settings['seed'] = None
        settings['speed'] = 1
        settings['gamma'] = 0.99
        settings['iterations'] = 1_000
        settings['print_details'] = False
        settings['n_reps'] = 10

        # variables (alter for each experiment)
        lr_list = [0.1, 0.01, 0.001]
        eta_list = [0.01, 0.1, 0.25]

        # setup experiments
        configs = []
        configs.append(lr_list)
        configs.append(eta_list)
        combinations = list(itertools.product(*configs))

        # print message
        print('Start experiments with REINFORCE algorithm ({} repetitions)'.format(settings['n_reps']))
        print('{} total experiments with combinations: {}'.format(len(combinations), combinations))

        Plot = LearningCurvePlot(title = 'REINFORCE: exploring learning rate and eta') 
        for comb in tqdm(combinations):
            print('combination running:{}'.format(comb))
            learning_curve, learning_curve_std = average_over_repetitions(exp=exp, settings=settings, comb=comb, smoothing_window=5)
            Plot.add_curve(learning_curve, label=r'lr:{}, eta:{}'.format(comb[0], comb[1]))
            Plot.add_confidence_interval(learning_curve, learning_curve_std)
        Plot.save('{}.png'.format(exp))

    elif exp == 'part1_actorcritic':
        # settings (fixed/constant during experiments)
        settings = dict()
        settings['rows'] = 7
        settings['columns'] = 7
        settings['max_steps'] = 250
        settings['max_misses'] = 10
        settings['observation_type'] = 'pixel'
        settings['seed'] = None
        settings['speed'] = 1
        settings['gamma'] = 0.99
        settings['iterations'] = 1_000
        settings['print_details'] = False
        settings['n_reps'] = 10
        settings['bootstrap'] = True
        settings['baseline_subtraction'] = True

        # variables (alter for each experiment)
        lr_list = [0.1, 0.01, 0.001]
        eta_list = [0.25, 0.1, 0.01]
        n_boot_list = [5]

        # setup experiments
        configs = []
        configs.append(lr_list)
        configs.append(eta_list)
        configs.append(n_boot_list)
        combinations = list(itertools.product(*configs))

        # print message
        print('Start experiments with Actor-Critic algorithm ({} repetitions)'.format(settings['n_reps']))
        print('{} total experiments with combinations: {}'.format(len(combinations), combinations))

        Plot = LearningCurvePlot(title='Actor-Critic: exploring learning rate and eta')
        for comb in tqdm(combinations):
            print('combination running:{}'.format(comb))
            learning_curve, learning_curve_std = average_over_repetitions(exp=exp, settings=settings, comb=comb, smoothing_window=5)
            Plot.add_curve(learning_curve, label=r'lr:{}, eta:{}'.format(comb[0], comb[1]))
            Plot.add_confidence_interval(learning_curve, learning_curve_std)
        Plot.save('{}.png'.format(exp))


if __name__ == '__main__':
    options = ['part1_reinforce', 'part1_actorcritic']

    for exp in options:
        print(exp)
        experiment(exp)

    # plot_smooth(exp,5)  # plot produced results with different smoothing window
