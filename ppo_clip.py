import torch
from torch import nn
from torch import optim
from tqdm import tqdm 
from catch import Catch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from helper import LearningCurvePlot, smooth, get_height
import os


##########################################################
# openAI paper ---> https://arxiv.org/pdf/1707.06347.pdf #
##########################################################


# hyperparameters
rows = 7
columns = 7
speed = 1.0
max_steps = 250
max_misses = 10
observation_type = 'pixel'      # 'vector'
seed = None

gamma = 0.99
iterations = 10
N_EPOCHS = 4
T_states, T_actions, T_probs, T_rewards, T_dones, T_values = list(), list(), list(), list(), list(), list()

def return_shuffle_batches(T_LIMIT, BATCH_SIZE):
    '''
    Return a list of lists that have BATCH_SIZE length and consist of random states shuffled in each
    one of them.
    param T_LIMIT:          timesteps for each policy update
    param BATCH_SIZE:       batch size of the lists
    '''
    index_list = [i for i in range(T_LIMIT)]
    np.random.shuffle(index_list)
    batches = [index_list[i:i+BATCH_SIZE] for i in range(0,T_LIMIT,BATCH_SIZE)]
    return batches

def updateDataStructures(state, action, probs, val, reward, done):
    '''
    It is called at each timestep in order to keep track of the states, actions, rewards, probabilities,
    critic output values and if done parameter. Generally, the maximum size of each of the six data structures
    can be as long as T_LIMIT length.
    param state:        state at the given timestep
    param action:       action chosen at the given timestep
    param probs:        probabilities
    param vals:         critic value
    param reward:       reward of the specific state
    param done:         True/False whether a state is terminal or not
    '''
    T_states.append(state)
    T_actions.append(action)
    T_probs.append(probs)
    T_values.append(val)
    T_rewards.append(reward)
    T_dones.append(done)
    

class ActorNetwork(nn.Module):
    '''
    Actor neural network model class. We build a simple NN for the actor consisting of 3 layers: an input layer, 
    a hidden layer with 128 neurons and an output layer. We use ReLU and Softmax activation functions for the 
    hidden and the output layer, respectively. We use Adam as our optimizer.
    '''
    def __init__(self, lr):
        super(ActorNetwork, self).__init__()

        if observation_type == 'pixel':
            self.n_inputs = rows*columns*2      # array of size [rows x columns x 2]
        elif observation_type == 'vector':
            self.n_inputs = 3                   # [x_paddle,x_lowest_ball,y_lowest_ball]
        self.n_outputs = 3                      # distribution over actions
        self.n_hidden = [128]

        self.actornetwork = nn.Sequential()
        # self.actornetwork.add_module("flatten", nn.Flatten(start_dim=0))
        self.actornetwork.add_module("input_hid", nn.Linear(self.n_inputs, self.n_hidden[0]))
        self.actornetwork.add_module("input_hidactiv", nn.ReLU(inplace=False))
        self.actornetwork.add_module("hid_output", nn.Linear(self.n_hidden[0], self.n_outputs))
        self.actornetwork.add_module("hid_outputactiv", nn.Softmax(dim=-1))

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def predict(self, state):
        action_probs = self.actornetwork(torch.FloatTensor(state))
        return action_probs


class CriticNetwork(nn.Module):
    '''
    Critic neural network model class. We build a simple NN for the critic consisting of 3 layers: an input layer, 
    a hidden layer with 128 neurons and an output layer with just one neuron. Although we use a ReLU activation function
    for the hidden layer, no activation is applied to the output of the model. We use Adam as our optimizer.
    '''
    def __init__(self, lr):
        super(CriticNetwork, self).__init__()

        if observation_type == 'pixel':
            self.n_inputs = rows*columns*2      # array of size [rows x columns x 2]
        elif observation_type == 'vector':
            self.n_inputs = 3                   # [x_paddle,x_lowest_ball,y_lowest_ball]
        self.n_outputs = 1                      # single critic_value without output activation 
        self.n_hidden = [128]

        self.criticnetwork = nn.Sequential()
        # self.criticnetwork.add_module("flatten", nn.Flatten(start_dim=0))
        self.criticnetwork.add_module("input_hid", nn.Linear(self.n_inputs, self.n_hidden[0]))
        self.criticnetwork.add_module("input_hidactiv", nn.ReLU(inplace=False))
        self.criticnetwork.add_module("hid_output", nn.Linear(self.n_hidden[0], self.n_outputs))

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def predict(self, state):
        output_value = self.criticnetwork(torch.FloatTensor(state))
        return output_value


def update_policy(actornetwork, criticnetwork, T_LIMIT, BATCH_SIZE, policy_clip):
    '''
    This functions is called every T_LIMIT steps to update the policy. The first thing we have to do 
    is to calculate the advantage for for this T_LIMIT steps using the data stored in the data structures.
    Then we find difference traces of length BATCH_SIZE with random states and update the policy. This is
    done for N_EPOCHS times.
    param actornetwork:     actor NN
    param criticnetwork:    critic NN
    param T_LIMIT:          timesteps for each policy update
    param BATCH_SIZE:       batch size of the lists
    param policy_clip:      policy clip or epsilon
    '''
    global T_states, T_actions, T_probs, T_values, T_rewards, T_dones
    for _ in range(N_EPOCHS):
        # 1. calculate the advantage (equations 11+12)
        advantage = np.zeros(T_LIMIT)
        lambda_smoothing = 1
        for t in range(T_LIMIT-1):
            current_adv = 0
            for k in range(t,T_LIMIT-1):
                # recall: critic_value of terminal state = 0 (convension in RL) 
                current_adv = current_adv + ((gamma*lambda_smoothing)**k)*(T_rewards[k] + gamma*T_values[k+1]*(1-int(T_dones[k])) - T_values[k])
            advantage[t] = current_adv
        advantage = torch.tensor(advantage)

        # 2. update in batches
        batches = return_shuffle_batches(T_LIMIT,BATCH_SIZE)
        for batch in batches:
            states = torch.tensor(np.array(T_states)[batch], dtype=torch.float)
            old_probs = torch.tensor(np.array(T_probs)[batch])
            actions = torch.tensor(np.array(T_actions)[batch])

            distribution = torch.distributions.Categorical(probs=actornetwork.predict(states.view(len(batch),-1)))
            critic_value = criticnetwork.predict(states.view(len(batch),-1))
            critic_value = torch.squeeze(critic_value)
            new_probs = distribution.log_prob(actions)
            
            r = new_probs.exp() / old_probs.exp()      # alternative: prob_ratio = (new_probs - old_probs).exp()
            term1 = r*advantage[batch] 
            term2 = torch.clamp(r, 1-policy_clip,1+policy_clip)*advantage[batch]
            actor_loss = -torch.min(term1, term2).mean()

            values = torch.tensor(T_values)
            critic_loss = (advantage[batch] + values[batch] - critic_value)**2
            critic_loss = critic_loss.mean()

            total_loss = actor_loss + 0.5*critic_loss
            actornetwork.optimizer.zero_grad()
            criticnetwork.optimizer.zero_grad()
            total_loss.backward()
            actornetwork.optimizer.step()
            criticnetwork.optimizer.step()

    T_states, T_actions, T_probs, T_rewards, T_dones, T_values = [], [], [], [], [], []


def run_experiments():
    '''
    Run different experiment with the PPO-clip algorithm. No parameters needed, everything is 
    done inside the function.
    '''

    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)
    
    combinations = [[0.001, 20, 10, 0.5, '1'],
                    [0.01, 20, 10, 0.5, '1'],[0.0001, 20, 10, 0.5, '1'],
                    [0.001, 20, 10, 0.2, '2'],[0.001, 20, 10, 0.8, '2'],
                    [0.001, 20, 5, 0.5, '3'],[0.001, 40, 5, 0.5, '3'],[0.001, 40, 10, 0.5, '3']]
    
    for comb in tqdm(combinations):
        lr = comb[0]
        T_LIMIT = comb[1]
        BATCH_SIZE = comb[2]
        policy_clip = comb[3]
        directory = comb[4]

        # set the environment and neural networks
        env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                    max_misses=max_misses, observation_type=observation_type, seed=seed)
        actornetwork = ActorNetwork(lr=lr)
        criticnetwork = CriticNetwork(lr=lr)

        rep_history = []
        for _ in tqdm(range(5)):            
            # run the algorithm
            rewards_per_episode = runPPOclip(env, actornetwork, criticnetwork, T_LIMIT, BATCH_SIZE, policy_clip)
            rep_history.append(rewards_per_episode)

        # store
        saved_dict = dict()
        saved_dict[str(comb)] = rep_history
        name = f'data_ppo/{directory}/combination_{lr}_{T_LIMIT}_{BATCH_SIZE}_{policy_clip}.pkl'
        a_file = open(name, 'wb')
        pickle.dump(saved_dict,a_file,pickle.HIGHEST_PROTOCOL)
        a_file.close()

        # make the figure
        if directory == '1':
            plot_title = r'PPO: exploring learning rate'
            legend_title = r'learning rate'
        elif directory == '2':
            plot_title = r'PPO: exploring policy clip'
            legend_title = r'policy clip'
        elif directory == '3':
            plot_title = r'PPO: exploring T-limit and batch size'
            legend_title = r'T-limit and batch size'
        plot = LearningCurvePlot(title=plot_title)

        central_path = os.path.dirname(__file__) + f'/data_ppo/{directory}'
        for f in os.listdir(central_path):
            target_f = central_path + '/' + f
            a_file = open(target_f, 'rb')
            x_dict = pickle.load(a_file)
            reward_results = x_dict[list(x_dict.keys())[0]]
            exp_settings = list(x_dict.keys())[0]
            a_file.close()

            learning_curve = np.mean(reward_results, axis=0)
            learning_curve = smooth(learning_curve, 2)  # additional smoothing
            learning_curve_std = np.std(reward_results, axis=0)  # calculate standard deviation for confidence interval

            tokens = f.split('_')
            if directory == '1':
                plot.add_curve(learning_curve, label=f'{tokens[1]}')
            elif directory == '2':
                plot.add_curve(learning_curve, label=f'{tokens[4]}')
            elif directory == '3':
                plot.add_curve(learning_curve, label=f'{tokens[2]} and {tokens[3]}')
            plot.add_confidence_interval(learning_curve, learning_curve_std)
            plot.add_hline(height=get_height(rows=7, speed=1))
            plot.save('experiment{}_win{}.png'.format(directory, 2), legend_title=legend_title)


def runPPOclip(env, actornetwork, criticnetwork, T_LIMIT, BATCH_SIZE, policy_clip):
    '''
    Function that runs PPO-clip. We collect a number episodes (#episodes=#iterations) and we update
    the policy every T_LIMIT steps.
    param env:          desired environment to run the algorithm (in our case: Catch environment) 
    param T_LIMIT:      timesteps to update the policy
    param BATCH_SIZE:   batch size variables
    param policy_clip:  policy clip or epsilon
    '''
    rewards_per_episode = []
    for _ in tqdm(range(iterations)):
        episode_steps = 0
        state = env.reset()
        done = False
        total_score = 0
        while not done:
            episode_steps += 1

            state_torch = torch.tensor([state], dtype=torch.float)
            distribution = torch.distributions.Categorical(probs=actornetwork.predict(state_torch.flatten()))

            action = distribution.sample()
            prob = distribution.log_prob(action).item()
            val = criticnetwork.predict(state_torch.flatten()).item()
        
            next_state, reward, done, _ = env.step(action)
            total_score += reward
            updateDataStructures(state, action, prob, val, reward, done)

            if episode_steps % T_LIMIT == 0:
                update_policy(actornetwork, criticnetwork, T_LIMIT, BATCH_SIZE, policy_clip)
            state = next_state
        rewards_per_episode.append(total_score)
    return rewards_per_episode


if __name__ == '__main__':
    # run_experiments()      # comment or uncomment to run experiments

    # initialize hyperparameters
    lr = 0.001
    T_LIMIT = 60
    BATCH_SIZE = 20
    policy_clip = 0.5

    # set the environment and neural networks
    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)
    actornetwork = ActorNetwork(lr=lr)
    criticnetwork = CriticNetwork(lr=lr)
        
    # run the algorithm
    rewards_per_episode = runPPOclip(env, actornetwork, criticnetwork, T_LIMIT, BATCH_SIZE, policy_clip)
    
    # make a graph
    x_axis = [i for i in range(1,iterations+1)]
    plt.plot(x_axis, rewards_per_episode)
    plt.show()
