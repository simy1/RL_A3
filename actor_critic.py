import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import catch
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from helper import *

class Actor(nn.Module):
    '''
    Actor neural network model class. We build a simple NN for the actor consisting of 3 layers: an input layer, 
    a hidden layer with 128 neurons and an output layer. We use ReLU and Softmax activation functions for the 
    hidden and the output layer, respectively.
    '''
    def __init__(self, n_input, n_actions):
        super(Actor, self).__init__()

        self.n_actions = n_actions
        self.n_hidden = 128

        self.actor_linear1 = nn.Linear(n_input, self.n_hidden)
        self.actor_linear2 = nn.Linear(self.n_hidden, n_actions)

    def forward(self, state):
        policy_probs = F.relu(self.actor_linear1(state))
        policy_probs = F.softmax(self.actor_linear2(policy_probs), dim=1)

        return policy_probs

class Critic(nn.Module):
    '''
    Critic neural network model class. We build a simple NN for the critic consisting of 3 layers: an input layer, 
    a hidden layer with 128 neurons and an output layer with just one neuron. Although we use a ReLU activation function
    for the hidden layer, no activation is applied to the output of the model.
    '''
    def __init__(self, n_input, n_actions):
        super(Critic, self).__init__()

        self.n_actions = n_actions
        self.n_hidden = 128

        self.critic_linear1 = nn.Linear(n_input, self.n_hidden)
        self.critic_linear2 = nn.Linear(self.n_hidden, 1) # only 1 state-value

    def forward(self, state):
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        return value

def run_episodes(env, actor, critic, actor_optimizer, critic_optimizer, bootstrap=True, baseline_subtraction=True,
                 n_episodes=1000, n_boot=1, gamma=0.99, entropy_reg_strength=0.01):
    '''
    Function for running the Actor-Critic algorithm. As we show below, for a number of iterations there are
    4 steps: (a) sample a trace, (b) compute the estimated cumulative reward, (c) compute and normalize the 
    loss of the actor and the critic, and (d) update the policy.
    param env:                      desired environment to un REINFORCE (in our case: Catch environment)     
    param actor:                    actor neural network model
    param critic:                   critic neural network model
    param actor_optimizer:          pytorch optimizer for actor NN (e.g. Adam)
    param critic_optimizer:         pytorch optimizer for critic NN (e.g. Adam)
    param bootstrap:                True/False flag for bootstrapping
    param baseline_subtraction:     True/False flag for baseline_subtraction
    param n_episodes:               integet number specifying the number of iterations
    param n_boot:                   number of steps for bootstrapping
    param gamma:                    gamma value (default value=0.99)
    param entropy_reg_strength:     variable for controlling entropy regularization
    '''
    rewards_per_episode = []
    epsilon = np.finfo(np.float32).eps.item()  # smallest possible value that won't get rounded off
    for episode in range(n_episodes):
        state = env.reset()
        state = torch.from_numpy(state.flatten()).float().unsqueeze(0) # convert to tensor
        terminal = False
        episode_rewards = []
        episode_values = []
        episode_log_action_prob = []
        episode_entropies = []
        while not terminal:
            # get the policy probability distribution and value
            policy_probs = actor.forward(state)
            value = critic.forward(state)
            episode_values.append(value)

            m = Categorical(policy_probs)
            action = m.sample()
            log_action_prob = m.log_prob(action)
            episode_log_action_prob.append(log_action_prob)
            entropy = -torch.sum(policy_probs * m.log_prob(torch.tensor([0,1,2])))
            episode_entropies.append(entropy)
            
            next_state, reward, terminal, _ = env.step(action)
            episode_rewards.append(reward)
            state = next_state
            state = torch.from_numpy(state.flatten()).float().unsqueeze(0) # convert to tensor

            if terminal:
                value = critic.forward(state)
                episode_values.append(value)
                break

        rewards_per_episode.append(np.sum(episode_rewards))     # track the performance

        # compute the estimated cumulative reward
        estimated_Q_values = []
        estimated_Q = 0
        if bootstrap and n_boot < len(episode_rewards):
            for i in range(len(episode_rewards)):
                estimated_Q = 0
                until_end = len(episode_rewards)-i-1 # amount of steps left until the end of the trace
                if until_end >= n_boot:
                    for n in range(n_boot):
                        estimated_Q += (gamma**n) * episode_rewards[i+n]
                    estimated_Q += gamma**(n_boot) * episode_values[i+n_boot]
                else:
                    for n in range(until_end):
                        estimated_Q += (gamma**n) * episode_rewards[i+n]
                    estimated_Q += gamma**(until_end) * episode_values[-1]
                estimated_Q_values.append(estimated_Q)
        else:
            for r in episode_rewards[::-1]: # sum backwards to make the computation more efficient
                estimated_Q = estimated_Q*gamma + r
                estimated_Q_values.insert(0, estimated_Q) # insert it at the start because we are working backwards
            estimated_Q_values = torch.tensor(estimated_Q_values)

        # compute the actor losses and critic losses
        actor_losses = []
        critic_losses = []
        advantages = []
        if baseline_subtraction:                        # compute and normalize the advantage
            for t in range(len(estimated_Q_values)):
                estimated_Q_t = estimated_Q_values[t]
                episode_values_t = episode_values[t]
                advantages.append(estimated_Q_t - episode_values_t)
                critic_losses.append(F.mse_loss(estimated_Q_t, episode_values_t))
            advantages = torch.tensor(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + epsilon)
            for t in range(len(estimated_Q_values)):
                actor_losses.append(-episode_log_action_prob[t] * advantages[t] - entropy_reg_strength * episode_entropies[t])
        else:                                           # normalize the estimated cumulative reward
            estimated_Q_values = torch.tensor(estimated_Q_values)
            estimated_Q_values = (estimated_Q_values - estimated_Q_values.mean()) / (estimated_Q_values.std() + epsilon)
            for t in range(len(estimated_Q_values)):
                estimated_Q_t = estimated_Q_values[t]
                actor_losses.append(-episode_log_action_prob[t] * estimated_Q_t - entropy_reg_strength * episode_entropies[t])
                critic_losses.append(F.mse_loss(estimated_Q_t, episode_values[t]))

        actor_losses = torch.stack(actor_losses).sum()
        critic_losses = torch.stack(critic_losses).sum()

        # update the policy/actor and value/critic networks
        actor_optimizer.zero_grad()
        actor_losses.backward(retain_graph=True)
        actor_optimizer.step()
        critic_optimizer.zero_grad()
        critic_losses.backward()
        critic_optimizer.step()

    return rewards_per_episode


if __name__ == '__main__':
    # initialize the environment and create the neural network
    rows = 7
    columns = 7
    speed = 1.0
    max_steps = 250
    max_misses = 10
    observation_type = 'pixel'  # 'vector'
    seed = None
    gamma = 0.99
    learning_rate = 0.01

    env = catch.Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)
    state = env.reset()
    n_input = len(state.flatten())
    state = torch.from_numpy(state.flatten()).float().unsqueeze(0) # convert to tensor
    n_actions = env.action_space.n
    actor = Actor(n_input=n_input, n_actions=n_actions)
    critic = Critic(n_input=n_input, n_actions=n_actions)
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

    response = ckeckCMD()
    if 'error' in response:
        printNotAcceptedCMD(response)
        exit()
    activate_BT = False
    activate_BS = False
    if 'BT' in response:
        activate_BT = True
    if 'BS' in response:
        activate_BS = True  

    # perform the algorithm
    print('Enabling Actor-Critic (Bootstrap={},Baseline Subtraction={})'.format(activate_BT,activate_BS))
    rewards_per_episode = run_episodes(env=env, actor=actor, critic=critic, actor_optimizer=actor_optimizer,
                                       critic_optimizer=critic_optimizer, bootstrap=activate_BT, baseline_subtraction=activate_BS,
                                       n_episodes=1000, n_boot=5, gamma=0.99, entropy_reg_strength=0.1)

    # make a plot
    plt.figure()
    plt.plot(range(len(rewards_per_episode)), rewards_per_episode)
    plt.savefig('Actor-Critic.png')
    plt.show()
