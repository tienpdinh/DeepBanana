[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, my directive was to train a Reinforcement Learning agent to collect yellow bananas and avoid blue bananas

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

2. Place the file in the same directory as the repository, and unzip (or decompress) the file. 

### Instructions

Start by creating a new virtual environment running under Python 3.6

```shell
$ virtualenv drl
```

Activate the environment using

```shell
$ source drl/bin/activate
```

Install the dependencies for the project

```shell
$ cd python/
$ pip install .
```

To start training the agent, run

```shell
$ python main.py <Double>

Usage: 
    python main.py <Double>
    Where <Double> can take either true or false, with true means using double DQN.
```

### Results
For regular DQN, the environment was solved after 503 episodes. For Double DQN, the environment was solved after 381 episodes!
