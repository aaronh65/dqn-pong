import gym

import torch
import torch.nn as nn
import torch.optim as optim

from common.atari_wrapper import *
from deepq.model import *
from deepq.learn import *

if __name__ == '__main__':
	# set device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# create environment
	env = gym.make("PongNoFrameskip-v4")
	env = make_env(env)

	steps_done = 0

	# train model
	train(env, 400, steps_done, device)

	# test model
	test(env, 1, device, render=True)