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
from reinforce import Model, runReinforceAlgo
from helper import LearningCurvePlot, smooth, saveResults, plot_smooth
from actor_critic import *


def average_over_repetitions(exp, settings, comb):
    '''
    Get the experimental setup and run for a number of repetitinos the experiment. Save the results at the end.
    param exp:          name of the experiment
    param settings:     parameters that stay the same 
    param comb:         parameters that we want to change
    '''

    reward_results = []  # Result array
    now = time.time()
    
    for _ in range(settings['n_reps']):  # Loop over repetitions
        if exp == 'reinforce':
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

        elif exp == 'actorcritic':
            env = Catch(rows=comb[3], columns=comb[4], speed=comb[6], max_steps=settings['max_steps'],
                    max_misses=settings['max_misses'], observation_type=comb[5], seed=settings['seed'])
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

    return


def experiment(exp):
    '''
    Setup of the different experiments so as to know which of the parameters and hyperparameters need to 
    stay the same and which should change in each experiment.
    param exp:      name of the experiment
    '''

    if exp == 'reinforce':
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

        for comb in tqdm(combinations):
            average_over_repetitions(exp, settings, comb)

    elif exp == 'actorcritic':
        # settings (fixed/constant during experiments)
        settings = dict()
        settings['max_steps'] = 250
        settings['max_misses'] = 10
        settings['seed'] = None
        settings['gamma'] = 0.99
        settings['iterations'] = 1_000
        settings['n_reps'] = 10
        settings['bootstrap'] = True
        settings['baseline_subtraction'] = True

        # variables (alter for each experiment)
        lr_list = [0.01]
        eta_list = [0.25]
        n_boot_list = [3]
        rows = [7]
        columns = [7]
        observation_type = ['pixel']
        speed = [1]

        # setup experiments
        configs = []
        configs.append(lr_list)
        configs.append(eta_list)
        configs.append(n_boot_list)
        configs.append(rows)
        configs.append(columns)
        configs.append(observation_type)
        configs.append(speed)
        combinations = list(itertools.product(*configs))

        # print message
        print('Start experiments with Actor-Critic algorithm ({} repetitions)'.format(settings['n_reps']))
        print('{} total experiments with combinations: {}'.format(len(combinations), combinations))

        for comb in tqdm(combinations):
            average_over_repetitions(exp, settings, comb)


if __name__ == '__main__':
    # =========== part 1 reinforce =============
    exp = 'part1_reinforce'
    plot_smooth(exp, False)  # plot produced results with different smoothing window


    # =========== part 1 actor critic =============
    ac_exps = ['AC_baseline tuning', 'AC_both tuning', 'AC_boot tuning']
    for exp in ac_exps:
        plot_smooth(exp, False)  # plot produced results with different smoothing window


    # =========== actor critic bootstrap =============
    bootstrap_exps = ['n_boot\AC_baseline', 'n_boot\AC_boot', 'n_boot\AC_both']
    for exp in bootstrap_exps:
        plot_smooth(exp, False)  # plot produced results with different smoothing window

    # =========== environment =============
    env_exp = ['vector', 'speed', 'sizes', 'combine', 'other']
    for exp in env_exp:
        print(exp)
        plot_smooth(exp, False)


