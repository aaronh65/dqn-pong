# -*- coding: utf-8 -*-
import random
import numpy as np
from itertools import count
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import gym

import math, time
from PIL import Image
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from common.atari_wrapper import *
from deepq.model import *
from deepq.replay_buffer import ReplayMemory
from collections import namedtuple
from autoencoder import AutoEncoder
from models import ResnetEncoder

import cv2
#from matplotlib import pyplot as plt

DEBUG = True

# hyperparameters
lr = 1e-4
if DEBUG:
    INITIAL_MEMORY = 200
    VAL_RATE = 1
    print('DEBUG')
else:
    INITIAL_MEMORY = 10000
    VAL_RATE = 10

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.02
EPS_DECAY = 250000
TARGET_UPDATE = 1000
MEMORY_SIZE = 10 * INITIAL_MEMORY

# program args
dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = Path(f"checkpoints/PongNoFrameskip-v4/{dt_str}")
save_dir.mkdir(parents=True,exist_ok=False) # do not overwrite if exists
(save_dir / 'images').mkdir()
encoder_path = "/home/aaron/workspace/vlr/dqn-pong/autoencoder/checkpoints/autoencoder/20210502_143858/epoch=9.ckpt"

config = {
        'lr': lr,
        'initial_memory': INITIAL_MEMORY,
        'batch_size': BATCH_SIZE,
        'gamma': GAMMA,
        'eps_start': EPS_START,
        'eps_end': EPS_END,
        'eps_decay': EPS_DECAY,
        'val_rate': VAL_RATE,
        'target_update': TARGET_UPDATE,
        'memory_size': MEMORY_SIZE,
        'time': dt_str,
        'encoder_path': encoder_path,
        'debug': DEBUG,
}

Transition = namedtuple('Transition',
			('state', 'action', 'next_state', 'reward'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize replay memory
memory = ReplayMemory(MEMORY_SIZE)

# create networks
#policy_net = DQNBase(n_actions=4).to(device)
#target_net = DQNBase(n_actions=4).to(device)
#target_net.load_state_dict(policy_net.state_dict())

#policy_net = DQNEncodedFeatures(512, n_actions=4).to(device)
#target_net = DQNEncodedFeatures(512, n_actions=4).to(device)
#target_net.load_state_dict(policy_net.state_dict())

policy_net = DQNEncodedLight(64, n_actions=4).to(device)
target_net = DQNEncodedLight(64, n_actions=4).to(device)
target_net.load_state_dict(policy_net.state_dict())


#TODO INIT ENCODER HERE
#weights_path = "/home/aaronhua/vlr/dqn-pong/autoencoder/checkpoints/autoencoder/20210502_143858/epoch=9.ckpt"
encoder = AutoEncoder.load_from_checkpoint(encoder_path).encoder.to(device)
encoder.eval()

#decoder = a_encoder.res_decoder
#decoder.eval()

transform = T.Compose([
            T.Resize((88, 88)),
            T.ToTensor()
            ])

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
            device=device, dtype=torch.bool)

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
    return loss

@torch.no_grad()
def get_state(obs, enc=False, debug=False):
    if not enc:
        state = np.array(obs)
        state = state.transpose((2, 0, 1))
        state = torch.from_numpy(state)
        return state.unsqueeze(0)
    else:
        obs = np.array(obs)
        obs = np.uint8(obs)
        frames = []
        if debug:
            print(obs.shape)
        for i in range(4):
            frame = Image.fromarray(obs[:, :, i*3:i*3+3])
            # frame.save(f'frame{i}.png')
            frames.append(transform(frame))
        frames = torch.stack(frames, 0).to(device)
        frames = encoder(frames)
        fram = [f for f in frames]
        fram = torch.cat(fram, 0)
        return fram.unsqueeze(0).detach().cpu()

def test(env, env_name, n_episodes, policy, device, render=True, restore=False, enc=False):
    #env = gym.wrappers.Monitor(env, './videos/' + 'dqn_breakout_video')
    env = gym.make(env_name)
    env = make_env(env, enc=enc)
    mean_total_reward = 0
    
    if restore:
        path = Path(f"checkpoints/{env_name}")
        ckpts = list(sorted(path.glob("*")))
        if len(ckpts) > 0:
            ckpt = ckpts[-1]
            policy = torch.load(str(ckpt))

    num_train_eps = len(list((save_dir / 'images').glob('*')))
    image_root = save_dir / 'images' / f'episode_{num_train_eps:06d}'
    image_root.mkdir()
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs, enc)
        total_reward = 0.0
        for t in count():
            action = policy(state.to(device)).max(1)[1].view(1,1)

            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)
            total_reward += reward

            if not done:
                next_state = get_state(obs, enc)
            else:
                next_state = None

            state = next_state

            if episode == 0: # save val images
                path = image_root / f'{t:06d}.png'
                img = np.uint8(np.array(obs))
                img = img[:,:,-3:]
                img = Image.fromarray(img).save(str(path))
    
            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break

        mean_total_reward += total_reward / n_episodes
    env.close()
    return mean_total_reward

def train(env, env_name, n_episodes, steps_done, device, render=False, enc=False, use_wandb=False):

    if use_wandb:
        config['ENV_NAME'] = env_name
        config['NUM_EPISODES'] = n_episodes
        wandb.init('vlr-project-dqn')
        wandb.config.update(config)

    path = Path(f"checkpoints/{env_name}")
    path.mkdir(parents=True,exist_ok=True)

    best_val_reward = -float('inf')
    for episode in tqdm(range(n_episodes)):
        obs = env.reset()
        
        if False:
            img = obs[:, :, :3] / 255
            #plt.figure(1)
            #plt.imsave('frame.png', img)

        state = get_state(obs, enc)
        
        total_reward = 0.0
        for t in count():
            metrics = {}
            action, steps_done = select_action(state, steps_done, device)

            if render:
                env.render()

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs, enc)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            memory.push(state, action.to(device), next_state, reward.to(device))
            state = next_state

            if steps_done > INITIAL_MEMORY: # BURN IN
                loss = optimize_model(device)
                metrics['loss'] = loss.item()

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                
            if done:
                metrics['train_reward'] = total_reward 
                #tqdm.write('Total steps: {} \t Episode: {}/{} \t Train reward: {}'.format(
                #    steps_done, episode, n_episodes, total_reward))

                if steps_done > INITIAL_MEMORY and episode % VAL_RATE == 0:
                    val_reward = test(env, env_name, 10, policy_net, device, render=render, enc=enc)
                    tqdm.write('Total steps: {} \t Episode: {}/{} \t Eval reward: {}'.format(
                        steps_done, episode, n_episodes, val_reward))
                    if val_reward >= best_val_reward:
                        torch.save(policy_net, str(save_dir / 'best.pt'))
                        best_val_reward = val_reward
                    metrics['val_reward'] = val_reward

                if use_wandb: # log before we go to next episode
                    wandb.log(metrics)
                break

            if use_wandb:
                wandb.log(metrics)
            
    env.close()
    return


