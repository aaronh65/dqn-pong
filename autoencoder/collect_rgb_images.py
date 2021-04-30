from datetime import datetime
import argparse
import gym
import cv2

from tqdm import tqdm
from pathlib import Path

def rollout(env, args, idx):

    save_root = args.save_root / f'episode_{idx:03d}' / 'rgb'
    save_root.mkdir(parents=True, exist_ok=False)

    done = False
    step = 0
    print(env.action_space)
    env.reset()

    while not done:
        obs = env.render(mode='rgb_array')
        new_obs, reward, done, info = env.step(env.action_space.sample())
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
        if args.show:
            cv2.imshow('env', obs)
            cv2.waitKey(10)

        path = str(save_root / f'{step:06d}.png')
        #print(path)
        cv2.imwrite(path, obs)
        step += 1

def main(args):
    env = gym.make(args.env)
    for i in tqdm(range(args.num_episodes)):
        rollout(env, args, i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_data', action='store_true')
    parser.add_argument('--save_root', type=str, default='data')
    #parser.add_argument('--env', type=str, default='Skiing-v0')
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--num_episodes', type=int, default=10)
    args = parser.parse_args()

    save_root = Path(args.save_root) / datetime.now().strftime("%Y%m%d_%H%M%S") 
    save_root.mkdir(parents=True, exist_ok=False)
    args.save_root = save_root
    main(args)
