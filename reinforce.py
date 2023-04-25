import sys
import numpy as np
import tensorflow as tf
import torch
from torch import nn

import matplotlib.pyplot as plt
from catch import Catch

class Reinforce_network(nn.Module):
    def __init__(self, num_inputs, num_hidden_nodes: int, num_actions: int = 3, learning_rate: float = 1e-3):
        print('Initializing Reinforce Network...')

        super(Reinforce_network, self).__init__()

        self.num_actions = num_actions

        self.layer1 = nn.Linear(in_features=num_inputs, out_features=num_hidden_nodes)
        self.layer2 = nn.Linear(in_features=num_hidden_nodes, out_features=num_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        result = torch.nn.functional.relu(self.layer1(state))
        result = torch.nn.functional.softmax(self.layer2(result), dim=1)
        return result

    def get_action(self, state):
        print('in get_action method')
        state = torch.from_numpy(state).float().unsqueeze(0)
        state = torch.autograd.Variable(state)

        probabilities = self.forward(state)
        probs = np.squeeze(probabilities.detach().numpy())
        action = np.random.choice(self.num_actions, p=probs)
        log_probability = torch.log(probabilities.squeeze(0)[action])

        return action, log_probability


def update_reinforce_network(reinforce_network, rewards, log_probabilities):
    print('updating reinfoce network')
    total_discounted_reward = 0  # R in course notes
    gradient = 0

    for r_t, log_probability in zip(rewards.reverse(), log_probabilities):
        total_discounted_reward = r_t + discount_factor * total_discounted_reward
        gradient += total_discounted_reward * log_probability  # TODO is this correct or do we need to take gradient / incorporate nabla somehow??


    reinforce_network.optimizer.zero_grad()  # set gradient to zero

    gradient = torch.stack(gradient)
    gradient.backward()  # compute gradient with respect to output

    reinforce_network.optimizer.step()  # perform single parameter update



def main_reinforce(num_episodes: int, episode_length: int, discount_factor: float, learning_rate: float, num_hidden_nodes: int):
    # Hyperparameters for environment, taken from catch.py test function
    # TODO add these as parameters to the function so we can vary them in part 2 of the assignment
    rows = 7
    columns = 7
    speed = 1.0
    max_steps = 250
    max_misses = 10
    observation_type = 'vector'  #'pixel' # 'vector'
    seed = None

    # Initialize environment
    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)
    s = env.reset()

    # initialize neural network for REINFORCE algorithm
    reinforce_network = Reinforce_network(num_inputs=env.observation_space.shape[0], num_hidden_nodes=num_hidden_nodes,
                                          num_actions=3, learning_rate=learning_rate)


    gradient = 0
    number_of_steps_taken = []  # record number of steps taken before failure
    all_rewards = []  # record all rewards during training

    for episode in range(num_episodes):  # loop 'for m in 1, ... M' in course notes
        s = env.reset()
        states = [s]
        actions = []
        log_probabilities = []
        rewards = []

        for step in range(episode_length):

            action, log_probability = reinforce_network.get_action(state=s)
            new_state, reward, done, _ = env.step(action)

            states.append(new_state)
            actions.append(action)
            log_probabilities.append(log_probability)
            rewards.append(reward)

            if done:
                update_reinforce_network(reinforce_network=reinforce_network, rewards=rewards, log_probability=log_probabilities)

                number_of_steps_taken.append(step)
                all_rewards.append(np.sum(rewards))

                break

            state = new_state







if __name__ == '__main__':
    # hyperparameters for reinforce neural network
    num_hidden_nodes = 20
    learning_rate = 1e-3

    # hyperparameters for REINFORCE algorithm
    discount_factor = 0.9  # gamma
    num_episodes = 6  # M
    episode_length = 10  # n
    num_hidden_nodes = 20

    main_reinforce(num_episodes=num_episodes, episode_length=episode_length, discount_factor=discount_factor,
                   learning_rate=learning_rate, num_hidden_nodes=num_hidden_nodes)




