#!/usr/bin/env python3
import sys
sys.path.append("..")

import os
import datetime
import argparse

import gym
import gym.spaces
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt

from ruamel.yaml import YAML

import torch
from torch import nn

import rslgym.algorithm.modules as rslgym_module
# import rslgym.algorithm.normalizer as normalizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument('-w', '--weight_dir', type=str, default='', help='path to trained')
    parser.add_argument('-i', '--iteration', type=int, default=0, help='algo iteration')
    parser.add_argument('-s', '--seconds', type=int, default=10, help='testing duration')

    args = parser.parse_args()
    weight_dir = args.weight_dir
    iteration = args.iteration

    save_path = os.path.join(weight_dir, 'testing_' + str(iteration), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(save_path)

    for file in os.listdir(weight_dir):
        if file.startswith('cfg'):
            cfg_abs_path = weight_dir + '/' + file

    # config
    cfg = YAML().load(open(cfg_abs_path, 'r'))

    # single env for testing
    test_env = gym.make(cfg['environment']['env_name'])
    test_env.seed(cfg['environment']['seed'])

    # https://github.com/openai/gym/issues/1925
    if cfg['environment']['record_video']:
        test_env = wrappers.Monitor(test_env, save_path, force=True, video_callable=lambda episode: True)

    obs_space = test_env.observation_space
    action_space = test_env.action_space

    actor_architecture = [64, 64]

    # obs_normalizer = normalizer.RunningMeanStd(shape=[obs_space.low.size])
    obs_normalizer = None

    torch.manual_seed(cfg['environment']['seed'])

    actor_net = nn.Sequential(
                rslgym_module.EmpiricalNormalization([obs_space.low.size]),
                rslgym_module.MLP(actor_architecture,
                    nn.LeakyReLU,
                    obs_space.low.size,
                    action_space.low.size)
                )

    actor = rslgym_module.Actor(actor_net,
                  rslgym_module.MultivariateGaussianDiagonalCovariance(action_space.low.size, 1.0),
                  obs_space.low.size,
                  action_space.low.size,
                  'cpu')

    # load actor weights

    snapshot = torch.load(weight_dir + '/snapshot' + str(iteration) + '.pt')
    actor.load_state_dict(snapshot['actor_state_dict'])

    test_steps = test_env.spec.max_episode_steps

    torch.manual_seed(args.seed)

    # containers for analysis
    actions = np.zeros(shape=(action_space.low.size, test_steps), dtype=np.float32)
    obs = np.zeros(shape=(obs_space.low.size, test_steps), dtype=np.float32)
    rews = np.zeros(shape=(1, test_steps), dtype=np.float32)

    ob = test_env.reset()
    ob = np.array(ob).reshape(1, -1).astype(np.float32)

    try:
        for i in range(test_steps):
            act = actor.noiseless_action(ob).cpu().detach().numpy()
            ob, r, done, info = test_env.step(act[0])
            ob = np.array(ob).reshape(1, -1).astype(np.float32)
            if cfg['environment']['render']:
                test_env.render()

            obs[:, i] = ob
            actions[:, i] = act
            rews[:, i] = r

            if done:
                break

    except KeyboardInterrupt:
        pass

    finally:
        if cfg['environment']['record_video']:
            # close video recording wrapper
            test_env.close()

        plt.figure()
        for i in range(action_space.low.size):
            plt.plot(actions[i, :], label='ac_' + str(i))
        plt.grid()
        plt.legend()

        plt.figure()
        for i in range(obs_space.low.size):
            plt.plot(obs[i, :], label='ob_' + str(i))
        plt.grid()
        plt.legend()

        plt.figure()
        plt.plot(rews[0, :], label='reward')
        plt.grid()
        plt.legend()

        plt.show(block=False)
        input('press [ENTER] to exit')


if __name__ == "__main__":
    main()
