# Reinforcement Learning - Policy-based RL


## Brief Description
This project shows how policy-based algorithms deal with the Catch Environment. Regarding our environment, we utilize a custamizable version of the Catch Environment (OpenAI GYM) produced by Thomas Moerland. The algorithms that are used are: (a) REINFORCE and (b) Actor-Critic. For the former, the implementation is straight forward. However, the Actor-Critic implementation makes use of some flags that enable us to run the corresponding algorithm with or without bootstrapping and baseline subtraction (thus, we have 4 distinct combinations). We enhance these algotithms by using entropy regularization. By doing so, we are able to maintain the distribution over action probabilities and avoid having one action which dominates the others. Therefore, we focus on how these algorithms, and especially the most powerful one (Actor-Critic with bootstapping and baseline subtraction), deal with the problem when hyperparameters, such as the learning rate, are modified. In addition, we investigate their performance when the environment alters, for example by enlarging the dimensions of the grid or changing the speed of dropping balls. Lastly, we implement and experiment with the state-of-the-art algorithm PPO-clip.


## Files
- requirements.txt: It saves a list of the modules and packages required for our project. Note: the whole project was implemented in a new virtual environment and therefore it contains only the required modules and packages for the purpose of the project with NO additional useless packages.

- catch.py: The custamizable version of the Catch Environment (OpenAI GYM) produced by Thomas Moerland.

- reinforce.py: This file hosts the implementatinon of the REINFORCE algorithm. We can run the program with the default (best) hyperparameters or change them. The program runs from the command line (see section 'How to run') and when it finishes it illustrates and saves the performance of the algorithm using a graph (x-axis: episodes, y-axis: reward per episode).

- actor_critic.py: It contains the code needed for the Actor-Critic algorithm. Check the section 'How to run' in order to enable or disable flags related to bootstrapping and baseline subtraction. When the code ends, it shows and saves a figure which depicts the performance of the algorithm for this problem (x-axis: episodes, y-axis: reward per episode).

- helper.py: This file consists of functions that consider to be helpful to either run one of the mentioned algorithms or carry out experiments. In addition, this python file can be used to vizualise results from experiments that are already performed. Some of the functions provided have to do with getting and controlling the input command line from terminal, others for saving the data for future use (such as plotting figures). 

- experiments.py: The experimental setup is hosted here. We specify which experiment we want to carry out (in the code) and then results for a number of repetitions are saved and illustrated with a graph. Some of the experiments are: playing around with the learning rate for REINFORCE and Actor-Critic and also changing parameters of the environment like the dimensions of the grid.

- ppo_clip.py: The PPO-clip state-of-the-art algorithm. In here, we have the implementation as well as an experimental setup to track the performance of the algorithm when altering some hyperparameters like learning rate, policy clip and other.


## How to run 
- Get the modules for the project:    
    - pip install -r requirements.txt

- Single run of the REINFORCE algorithm:    
    - python reinforce.py

- Single run of the Actor-Critic algorithm:    
    - python actor_critic.py
    - python actor_critic.py --bootstr
    - python actor_critic.py --basesub
    - python actor_critic.py --bootstr --basesub

- Single run of the experiments.py to produce all of our results:    
    - python experiments.py

- Single run of the PPO-clip algorithm:    
    - python ppo_clip.py

## License
MIT License
