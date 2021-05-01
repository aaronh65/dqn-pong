import gym
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from common.atari_wrapper import *
from deepq.model import *
from deepq.learn import *

ENV_NAME = "PongNoFrameskip-v4"
# ENV_NAME = "Breakout-v0"
#NUM_EPISODES = 400
NUM_EPISODES = 10000
RENDER=False

TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
USE_WANDB = True
ENCODER = True
if __name__ == '__main__':

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create environment
    env = gym.make(ENV_NAME)
    env = make_env(env, enc=ENCODER)

    steps_done = 0

    # train model
    train(env, ENV_NAME, NUM_EPISODES, steps_done, device, render=RENDER, enc=ENCODER)

    # test model
    test(env, ENV_NAME, 100, policy_net, device, render=RENDER, enc=ENCODER)




