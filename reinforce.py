#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custamizable Catch Environment
By Thomas Moerland
Leiden University, The Netherlands
2022

Extended from Catch environment in Behavioural Suite: https://arxiv.org/pdf/1908.03568.pdf
"""

import matplotlib
matplotlib.use('TkAgg') #'Qt5Agg') # 'TkAgg'
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from gym import spaces   

#-------------------------------------------------------------------
# my imports
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from catch import Catch
#-------------------------------------------------------------------

class Model():
    def __init__(self, observation_type, rows, columns):
        if observation_type == 'pixel':
            self.n_inputs = rows*columns*2          # array of size [rows x columns x 2]
        elif observation_type == 'vector':
            self.n_inputs = 3                       # [x_paddle,x_lowest_ball,y_lowest_ball]
        self.n_outputs = 3                          # distribution over actions
        self.n_hidden = [128]

        self.neuralnetwork = nn.Sequential()
        self.neuralnetwork.add_module("flatten", nn.Flatten(start_dim=0))
        self.neuralnetwork.add_module("input_hid", nn.Linear(self.n_inputs, self.n_hidden[0]))
        self.neuralnetwork.add_module("input_hidactiv", nn.ReLU())
        self.neuralnetwork.add_module("hid_output", nn.Linear(self.n_hidden[0], self.n_outputs))
        self.neuralnetwork.add_module("hid_outputactiv", nn.Softmax(dim=-1))

    def predict(self, state):
        action_probs = self.neuralnetwork(torch.FloatTensor(state))
        return action_probs


def generateTrace(env, model, entropy_term):
    states_trace,actions_trace,rewards_trace = [],[],[]
    win_loss_ratio = [0,0]
    sum_rewards = 0
    done = False
    state = torch.tensor(env.reset(), dtype=torch.float)

    while not done:
        probs = model.predict(state)
        distribution = torch.distributions.Categorical(probs=probs)
        action = distribution.sample().item()
        state_next,reward,done,_ = env.step(action)
        # env.render(step_pause)          # visualize the procedure during training

        if reward == 0:
            pass
        elif reward == 1:
            win_loss_ratio[0] += 1
        elif reward == -1:
            win_loss_ratio[1] += 1
        sum_rewards += reward

        # for entropy regularization
        # entropy = -np.sum(np.mean(np.array(probs)) * np.log(np.array(probs)))
        smoothing_value = 1.0e-10       # smoothing value to avoid calculating log of zero (=infinity)
        probs = probs.detach().numpy()
        smoothed_probs = np.where(probs != 0, probs, smoothing_value)
        entropy = -np.sum(np.array(smoothed_probs) * np.log(np.array(smoothed_probs)))
        entropy_term += entropy

        states_trace.append(state)
        actions_trace.append(torch.tensor(action, dtype=torch.int))
        rewards_trace.append(reward)

        state = torch.tensor(state_next, dtype=torch.float)

    return states_trace,actions_trace,rewards_trace, win_loss_ratio, entropy_term, sum_rewards


def compute_discount_rewards(rewards_list, gamma):
    disc_rewards = []
    for t in range(len(rewards_list)):
        G = 0.0
        for k,r in enumerate(rewards_list[t:]):
            G += (gamma**k)*r
        disc_rewards.append(G)

    # normalization
    disc_rewards = torch.tensor(disc_rewards)
    disc_rewards = (disc_rewards - disc_rewards.mean()) / (disc_rewards.std() + np.finfo(np.float32).eps.item())

    return disc_rewards


def update_policy(states_list, actions_list, g_list, model, optimizer, eta, entropy_term, print_details):
    loss_stored = []
    for state, action, G in zip(states_list, actions_list, g_list):
        probs = model.predict(state)
        distribution = torch.distributions.Categorical(probs=probs)
        log_prob = distribution.log_prob(action)

        loss = - log_prob*G
        loss = loss - eta * entropy_term

        if print_details:
            print('probs = {}'.format(probs))
            print('log_prob = {} // G = {} '.format(log_prob, G))

        loss_stored.append(loss)

        optimizer.zero_grad()
        loss.backward()         # calculate gradients
        optimizer.step()        # apply gradients

    if print_details:
        print('losses found: ', loss_stored)


def runReinforceAlgo(env=None, model=None, optimizer=None, gamma=0.9, iterations=1000, eta=0.001, print_details=False):
    entropy_term = 0
    rewards_per_episode = []
    # print('Enabling REINFORCE algorithm . . .')
    for _ in range(iterations):
        # initialization
        states_list, actions_list, rewards_list = [], [], []

        # generate a trace
        states_list, actions_list, rewards_list, win_loss_ratio, entropy_term, sum_rewards = generateTrace(env=env, model=model, entropy_term=entropy_term)
        rewards_per_episode.append(sum_rewards)
        if print_details:
            print('Trace genrated . . . Done --> Length of trace: {} - win/loss ratio: {}/{}'.format(len(states_list), win_loss_ratio[0], win_loss_ratio[1]))

        # compute discount rewards 
        # print('Computing G . . .')
        g_list = []
        g_list = compute_discount_rewards(rewards_list=rewards_list, gamma=gamma)

        # update the policy
        # print('Updating the policy . . .')
        update_policy(states_list=states_list, actions_list=actions_list, g_list=g_list, model=model, optimizer=optimizer, eta=eta, entropy_term=entropy_term, print_details=print_details)

    return rewards_per_episode
    

if __name__ == '__main__':
    # Set hyperparameters 
    rows = 7
    columns = 7
    speed = 1.0
    max_steps = 250
    max_misses = 10
    observation_type = 'pixel'      # 'vector'
    seed = None
    gamma = 0.99
    lr = 0.001

    # Initialize environment, model, optimizer
    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)
    s = env.reset()
    # step_pause = 0.2 # the pause between each plot
    # env.render(step_pause) 

    model = Model(observation_type, rows, columns)

    optimizer = optim.Adam(model.neuralnetwork.parameters(), lr=lr)
    
    # Run REINFORCE
    iterations = 1_000
    eta = 0.25
    print_details = False
    rewards_per_episode = runReinforceAlgo(env=env,
                                           model=model, 
                                           optimizer=optimizer,
                                           gamma=gamma,
                                           iterations=iterations, 
                                           eta=eta,
                                           print_details=print_details)
    
    x_axis = [i for i in range(1,iterations+1)]
    plt.plot(x_axis, rewards_per_episode)
    plt.savefig('example_graph.png')
    plt.show()