import sys
import numpy as np
import tensorflow as tf
from catch import Catch

class Reinforce_network():
    def __init__(self, input_shape, num_hidden_nodes: int, num_actions: int = 3, learning_rate: float = 1e-3):
        print('Initializing Reinforce Network...')
        self.num_actions = num_actions

        self.model = tf.keras.models.Sequential([
            # tf.keras.layers.Flatten(input_shape=input_shape),
            # tf.keras.layers.Flatten(input_shape=[7,7,2]),
            tf.keras.layers.Dense(24, activation='relu'), #, input_shape=input_shape),
            tf.keras.layers.Dense(num_hidden_nodes, activation='relu'),
            tf.keras.layers.Dense(num_actions, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

    def get_action(self, state):
        print('get_action')

        print(s.shape)

        action = self.model.predict(state)
        # action = self.model(state, training=False)
        print(action, action.shape)

        print(self.model.summary())

        print(self.model.output_shape)

        return action

    def update(self):
        pass





# def reinforce(learning_rate, num_episodes, episode_lengths, epsilon = 0.01):
#     theta = 0  # TODO
#     converged = False
#
#     while not converged:
#         grad = 0
#
#         gradient = 0
#         for m in range(num_episodes):
#             R = 0
#             states = []
#             actions = []
#             rewards = []
#
#             # TODO
#
#             for t in range(episode_lengths, -1, -1):
#                 R = rewards[t] + discount_factor * R
#                 grad += R  # TODO
#
#         # update
#         theta = theta + learning_rate * grad
#
#         if grad < epsilon:
#             converged = True
#
#
#
#     # ======= initialize ========




def main_reinforce():
    # Hyperparameters for environment, taken from catch.py test function
    rows = 7
    columns = 7
    speed = 1.0
    max_steps = 250
    max_misses = 10
    observation_type = 'vector'  #'pixel' # 'vector'
    seed = None

    # Initialize environment and Q-array
    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)
    s = env.reset()
    reinforce_network = Reinforce_network(input_shape=s.shape, num_hidden_nodes=10)

    for episode in range(num_episodes):
        s = env.reset()
        states = [s]
        actions = []
        rewards = []

        for step in range(max_steps):
            action = Reinforce_network.get_action(state=s)
            new_state, reward, done, _ = env.step(action)





if __name__ == '__main__':
    # hyperparameters for REINFORCE
    learning_rate = 0.1  # eta
    discount_factor = 0.9  # gamma
    num_episodes = 6  # M
    episode_lengths = 10  # n

    # Hyperparameters for environment, taken from catch.py test function
    rows = 7
    columns = 7
    speed = 1.0
    max_steps = 250
    max_misses = 10
    observation_type = 'pixel'  #'pixel' # 'vector'
    seed = None

    # Initialize environment and Q-array
    env = Catch(rows=rows, columns=columns, speed=speed, max_steps=max_steps,
                max_misses=max_misses, observation_type=observation_type, seed=seed)
    s = env.reset()

    print(s.flatten().shape)
    print('Ball:')
    print(s[:,:,0].T)
    print('=====')
    print('Paddle')
    print(s[:,:,1].T)


    test = Reinforce_network(input_shape=s.shape,
                             num_hidden_nodes=3)

    # print(test.model.summary())

    action = test.get_action(state=tf.Tensor([s], dtype=int) )

