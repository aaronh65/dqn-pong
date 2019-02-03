# -*- coding: utf-8 -*-
import random
import numpy as np
from itertools import count
import gym

import math, time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from common.atari_wrapper import *
from deepq.model import *
from deepq.replay_buffer import ReplayMemory
from collections import namedtuple

# hyperparameters
lr = 1e-4
INITIAL_MEMORY = 10000
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.02
EPS_DECAY = 1000000
TARGET_UPDATE = 1000
RENDER = False
MEMORY_SIZE = 10 * INITIAL_MEMORY

Transition = namedtuple('Transion',
			('state', 'action', 'next_state', 'reward'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize replay memory
memory = ReplayMemory(MEMORY_SIZE)

# create networks
policy_net = DQN(n_actions=4).to(device)
target_net = DQN(n_actions=4).to(device)
target_net.load_state_dict(policy_net.state_dict())

# setup optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=lr)

def select_action(state, steps_done, device):
	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END)* \
		math.exp(-1. * steps_done / EPS_DECAY)
	steps_done += 1
	if sample > eps_threshold:
		with torch.no_grad():
			return policy_net(state.to(device)).max(1)[1].view(1,1), steps_done
	else:
		return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long), steps_done

def optimize_model(device):
	if len(memory) < BATCH_SIZE:
		return
	transitions = memory.sample(BATCH_SIZE)
	"""
	zip(*transitions) unzips the transitions into
	Transition(*) creates new named tuple
	batch.state - tuple of all the states (each state is a tensor)
	batch.next_state - tuple of all the next states (each state is a tensor)
	batch.reward - tuple of all the rewards (each reward is a float)
	batch.action - tuple of all the actions (each action is an int)
	"""
	batch = Transition(*zip(*transitions))

	actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action)))
	rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward)))

	non_final_mask = torch.tensor(
		tuple(map(lambda s: s is not None, batch.next_state)),
		device=device, dtype=torch.uint8)

	non_final_next_states = torch.cat([s for s in batch.next_state
					if s is not None]).to(device)

	state_batch = torch.cat(batch.state).to(device)
	action_batch = torch.cat(actions)
	reward_batch = torch.cat(rewards)

	state_action_values = policy_net(state_batch).gather(1, action_batch)

	next_state_values = torch.zeros(BATCH_SIZE, device=device)
	next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch

	loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

	optimizer.zero_grad()
	loss.backward()
	for param in policy_net.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()

def get_state(obs):
	state = np.array(obs)
	state = state.transpose((2, 0, 1))
	state = torch.from_numpy(state)
	return state.unsqueeze(0)

def train(env, n_episodes, steps_done, device, render=False):
	for episode in range(n_episodes):
		obs = env.reset()
		state = get_state(obs)
		total_reward = 0.0
		for t in count():
			action, steps_done = select_action(state, steps_done, device)

			if render:
				env.render()

			obs, reward, done, info = env.step(action)

			total_reward += reward

			if not done:
				next_state = get_state(obs)
			else:
				next_state = None

			reward = torch.tensor([reward], device=device)

			memory.push(state, action.to(device), next_state, reward.to(device))
			state = next_state

			if steps_done > INITIAL_MEMORY:
				optimize_model(device)

				if steps_done % TARGET_UPDATE == 0:
					target_net.load_state_dict(policy_net.state_dict())

			if done:
				break
		if episode % 20 == 0:
			print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t, total_reward))
	env.close()
	torch.save(policy_net, "dqn_pong_model")
	return

def test(env, n_episodes, policy, device, render=True):
	env = gym.wrappers.Monitor(env, './videos/' + 'dqn_pong_video')
	policy_net = torch.load("dqn_pong_model")
	for episode in range(n_episodes):
		obs = env.reset()
		state = get_state(obs)
		total_reward = 0.0
		for t in count():
			action = policy(state.to(device)).max(1)[1].view(1,1)

			if render:
				env.render()
				time.sleep(0.02)

			obs, reward, done, info = env.step(action)

			total_reward += reward

			if not done:
				next_state = get_state(obs)
			else:
				next_state = None

			state = next_state

			if done:
				print("Finished Episode {} with reward {}".format(episode, total_reward))
				break

		env.close()
	return