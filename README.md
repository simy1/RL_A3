# Reinforcement Learning - Policy-based RL


## Brief Description
This project shows how policy-based algorithms deal with the Catch Environment. Regarding our environment, we utilize a custamizable version of the Catch Environment (OpenAI GYM) produced by Thomas Moerland. The algorithms that are used are: (a) REINFORCE and (b) Actor-Critic. For the former, the implementation is straight forward. However, the Actor-Critic implementation makes use of some flags that enable us to run the corresponding algorithm with or without bootstrapping and baseline subtraction (thus, we have 4 distinct combinations). We enhance these algotithms by using entropy regularization. By doing so, we are able to maintain the distribution over action probabilities and avoid having one action which dominates the others. Therefore, we focus on how these algorithms, and especially the most powerful one (Actor-Critic with bootstapping and baseline subtraction), deal with the problem when hyperparameters, such as the learning rate, are modified. In addition, we investigate their performance when the environment alters, for example by enlarging the dimensions of the grid or changing the speed of dropping balls. Lastly, we [PPO here].


## Files
- requirements.txt: It saves a list of the modules and packages required for our project. Note: the whole project was implemented in a new virtual environment and therefore it contains only the required modules and packages for the purpose of the project with NO additional useless packages.

- catch.py: The custamizable version of the Catch Environment (OpenAI GYM) produced by Thomas Moerland.

- reinforce.py: This file hosts the implementatinon of the REINFORCE algorithm. We can run the program with the default (best) hyperparameters or change them. The program runs from the command line (see section 'How to run') and when it finishes it illustrates and saves the performance of the algorithm using a graph (x-axis: episodes, y-axis: reward per episode).

- actor_critic.py: This file contains the code needed for the Actor-Critic algorithm. Check the section 'How to run' in order to enable or disable flags related to bootstrapping and baseline subtraction. When the code ends, it shows and saves a figure which depicts the performance of the algorithm for this problem (x-axis: episodes, y-axis: reward per episode).

- helper.py: This file consists of functions that consider to be helpful to either run one of the mentioned algorithms or carry out experiments. In addition, this python file can be used to vizualise results from experiments that are already performed. Some of the functions provided have to do with getting and controlling the input command line from terminal, others for saving the data for future use (such as plotting figures). 

- experiments.py:

- ppo_clip.py: [PPO here].


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
    

## License
MIT License
