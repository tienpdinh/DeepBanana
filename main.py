from unityagents import UnityEnvironment
from dqn_agent import Agent
from collections import deque
import torch
import matplotlib.pyplot as plt
import numpy as np

import sys

# Handle arguments passing into the file
if len(sys.argv) < 2 or (sys.argv[1] != 'true' and sys.argv[1] != 'false'):
    print('Usage: ')
    print('\tpython main.py <Double>')
    print('\tWhere <Double> can take either true or false, with true means using double DQN.')
    exit(1)

DDQN = False
if sys.argv[1] == 'true':
    DDQN = True

# load the environment
env = UnityEnvironment(file_name="Banana_Linux_NoVis/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

# define the agent
agent = Agent(state_size, action_size, 0, DDQN)

def train(episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for episode in range(1, episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for _ in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

def main():
    scores = train()   
    # Plot and save the rewards
    plt.clf()
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.savefig('reward.png')
    env.close()

if __name__ == '__main__':
    main()