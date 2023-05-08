#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib

matplotlib.use('TkAgg')  # 'Qt5Agg') # 'TkAgg'
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from gym import spaces
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from catch import Catch


class Model():
    '''
    Neural network model class. We build a simple NN consisting of 4 layers: a layer that flattens the 
    input, an input layer, a hidden layer with 128 neurons and an output layer. We use ReLU and Softmax
    activation functions for the hidden and the output layer, respectively.
    '''

    def __init__(self, observation_type, rows, columns):
        if observation_type == 'pixel':
            self.n_inputs = rows * columns * 2  # array of size [rows x columns x 2]
        elif observation_type == 'vector':
            self.n_inputs = 3  # [x_paddle,x_lowest_ball,y_lowest_ball]
        self.n_outputs = 3  # distribution over actions
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
    '''
    Generate a trace. Collect triplets of state-action-reward until a terminal condition is met.
    At every step, we use entropy regularization to maintain our distribution and avoid having an
    action that dominates the others.
    param env:              environment used
    param model:            neural network model
    param entropy_term:     variable for adding entropies
    '''
    states_trace, actions_trace, rewards_trace = [], [], []
    win_loss_ratio = [0, 0]
    sum_rewards = 0
    done = False
    state = torch.tensor(env.reset(), dtype=torch.float)

    while not done:
        probs = model.predict(state)
        distribution = torch.distributions.Categorical(probs=probs)
        action = distribution.sample().item()
        state_next, reward, done, _ = env.step(action)
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
        smoothing_value = 1.0e-10  # smoothing value to avoid calculating log of zero (=infinity)
        probs = probs.detach().numpy()
        smoothed_probs = np.where(probs != 0, probs, smoothing_value)
        entropy = -np.sum(np.array(smoothed_probs) * np.log(np.array(smoothed_probs)))
        entropy_term += entropy

        states_trace.append(state)
        actions_trace.append(torch.tensor(action, dtype=torch.int))
        rewards_trace.append(reward)

        state = torch.tensor(state_next, dtype=torch.float)

    return states_trace, actions_trace, rewards_trace, win_loss_ratio, entropy_term, sum_rewards


def compute_discount_rewards(rewards_list, gamma):
    '''
    Compute the discount rewards and then apply normalization.
    param rewards list:     list of rewards for the current trace
    param gamma:            gamma value 
    '''
    disc_rewards = []
    for t in range(len(rewards_list)):
        G = 0.0
        for k, r in enumerate(rewards_list[t:]):
            G += (gamma ** k) * r
        disc_rewards.append(G)

    # normalization
    disc_rewards = torch.tensor(disc_rewards)
    disc_rewards = (disc_rewards - disc_rewards.mean()) / (disc_rewards.std() + np.finfo(np.float32).eps.item())

    return disc_rewards


def update_policy(states_list, actions_list, g_list, model, optimizer, eta, entropy_term, print_details):
    '''
    Update the policy based on a given trace and the corresponding discount rewards.
    param states_list:          list of states of the trace
    param actions_list:         list of actions of the trace
    param g_list:               list of discount rewards
    param model:                neural network model
    param optimizer:            pytorch optimizer (e.g. Adam) 
    param eta:                  variable for controlling entorpy regularization
    param entropy term:         entropies collected
    param print_details:        True/False whether we need to print helpful messages in the console
    '''
    loss_stored = []
    for state, action, G in zip(states_list, actions_list, g_list):
        probs = model.predict(state)
        distribution = torch.distributions.Categorical(probs=probs)
        log_prob = distribution.log_prob(action)

        loss = - log_prob * G
        loss = loss - eta * entropy_term

        if print_details:
            print('probs = {}'.format(probs))
            print('log_prob = {} // G = {} '.format(log_prob, G))

        loss_stored.append(loss)

        optimizer.zero_grad()
        loss.backward()  # calculate gradients
        optimizer.step()  # apply gradients

    if print_details:
        print('losses found: ', loss_stored)


def runReinforceAlgo(env=None, model=None, optimizer=None, gamma=0.9, iterations=1000, eta=0.001, print_details=False,
                     calculate_variance=False):
    '''
    Function for running the REINFORCE algorithm. For a number of iterations we do 3 things: (a) generate a trace, 
    (b) compute the discount rewards and (c) update the policy.
    param env:              desired environment to un REINFORCE (in our case: Catch environment)
    param model:            neural network model
    param optimizer:        pytorch optimizer (e.g. Adam)
    param gamma:            gamma value (default value=0.9)
    param iterations:       integet number specifying the number of iterations
    param eta:              variable for controlling entropy regularization
    param print_details:    True/False whether we need to print helpful messages in the console 
    '''
    entropy_term = 0
    rewards_per_episode = []
    # print('Enabling REINFORCE algorithm . . .')

    if calculate_variance:
        layer_means = np.zeros((4, iterations))

    for iteration in range(iterations):
        # initialization
        states_list, actions_list, rewards_list = [], [], []

        # generate a trace
        states_list, actions_list, rewards_list, win_loss_ratio, entropy_term, sum_rewards = generateTrace(env=env,
                                                                                                           model=model,
                                                                                                           entropy_term=entropy_term)
        rewards_per_episode.append(sum_rewards)
        if print_details:
            print('Trace genrated . . . Done --> Length of trace: {} - win/loss ratio: {}/{}'.format(len(states_list),
                                                                                                     win_loss_ratio[0],
                                                                                                     win_loss_ratio[1]))

        # compute discount rewards 
        # print('Computing G . . .')
        g_list = []
        g_list = compute_discount_rewards(rewards_list=rewards_list, gamma=gamma)

        if calculate_variance:
            state_dict = model.neuralnetwork.state_dict()
            for i, (layer_name, layer_params) in enumerate(state_dict.items()):
                layer = np.array(layer_params)
                layer_means[i, iteration] = np.mean(layer)

        # update the policy
        # print('Updating the policy . . .')
        update_policy(states_list=states_list, actions_list=actions_list, g_list=g_list, model=model,
                      optimizer=optimizer, eta=eta, entropy_term=entropy_term, print_details=print_details)

    if calculate_variance:
        return layer_means

    return rewards_per_episode


def calculate_reinforce_variance(env=None, model=None, optimizer=None, gamma=0.9, iterations=1000, eta=0.001, print_details=False,
                                 num_runs: int = 10):

    num_layers = 4
    all_layer_means = np.zeros((num_layers, iterations, num_runs))
    layer_names = ['input_hid.weight', 'input_hid.bias', 'hid_output.weight', 'hid_output.bias']
    layer_names = ['input layer: weights', 'input layer: bias', 'output layer: weights', 'output layer: bias']

    for run in tqdm(range(num_runs)):
        layer_means = runReinforceAlgo(env=env,
                         model=model,
                         optimizer=optimizer,
                         gamma=gamma,
                         iterations=iterations,
                         eta=eta,
                         print_details=print_details,
                         calculate_variance=True)

        all_layer_means[:, :, run] = layer_means

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")

    for l in range(num_layers):
        layer_weight_av = np.mean(all_layer_means[l, :, :], axis=1)
        layer_weight_std = np.std(all_layer_means[l, :, :], axis=1)

        ax.plot(layer_weight_av, label=layer_names[l])
        ax.fill_between(x=range(iterations), y1=layer_weight_av-layer_weight_std, y2=layer_weight_av+layer_weight_std, alpha=0.2)

    ax.legend(bbox_to_anchor=(1, 1), title='Layer', alignment='left')
    ax.set_xlabel('Iteration', fontsize=13)
    ax.set_ylabel('Average Weight', fontsize=13)
    ax.set_title(f'Average Model Weights over {num_runs} independent iterations', fontsize=15)

    fig.savefig('variance.png', dpi=300)
    fig.savefig('variance.svg', dpi=300)
    # fig.show()


if __name__ == '__main__':
    # Set hyperparameters 
    rows = 7
    columns = 7
    speed = 1.0
    max_steps = 250
    max_misses = 10
    observation_type = 'pixel'  # 'vector'
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
    iterations = 10  # 1_000
    eta = 0.25
    print_details = False
    rewards_per_episode = runReinforceAlgo(env=env,
                                           model=model,
                                           optimizer=optimizer,
                                           gamma=gamma,
                                           iterations=iterations,
                                           eta=eta,
                                           print_details=print_details)

    # make a plot
    x_axis = [i for i in range(1, iterations + 1)]
    plt.plot(x_axis, rewards_per_episode)
    plt.xlabel("Episodes")
    plt.ylabel("Total reward per episode")
    plt.title('REINFORCE')
    plt.savefig('REINFORCE.png')
    plt.show()

    # ======================================================================================================
    # does reinforce suffer from high variance?
    calculate_reinforce_variance(env=env, model=model, optimizer=optimizer, gamma=gamma, iterations=1000,
                                 eta=eta, print_details=print_details, num_runs=10)

