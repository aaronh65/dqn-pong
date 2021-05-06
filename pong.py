import gym
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from common.atari_wrapper import *
from deepq.model import *
from deepq.learn import *

ENV_NAME = "PongNoFrameskip-v4"
NUM_STEPS = 1e6
RENDER = False
ENCODER = True
USE_WANDB = True

if __name__ == '__main__':

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create environment
    env = gym.make(ENV_NAME)
    env = make_env(env, enc=ENCODER)

    steps_done = 0

    # train model
    train(env, ENV_NAME, NUM_EPISODES, steps_done, device, render=RENDER, enc=ENCODER, use_wandb=USE_WANDB)

    # test model
    #test(env, ENV_NAME, 100, policy_net, device, render=RENDER, enc=ENCODER)




