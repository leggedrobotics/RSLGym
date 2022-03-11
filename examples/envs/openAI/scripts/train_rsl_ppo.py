#!/usr/bin/env python3
import sys
sys.path.append("..")

import os
import argparse
import functools
import time

import gym
import gym.spaces
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from ruamel.yaml import YAML

import torch
from torch import nn

import rslgym.algorithm.modules as rslgym_module
from rslgym.algorithm.agents.ppo import PPO
from rslgym.algorithm.utils import ConfigurationSaver

from multithread_vector_env import MultiprocessVectorEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_name', type=str, default='/cfg.yaml', help='configuration file')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id (-1 for cpu)')
    args = parser.parse_args()
    cfg_name = args.cfg_name

    device = args.gpu if args.gpu > 0 else 'cpu'

    task_path = os.path.dirname(os.path.realpath(__file__))
    cfg_abs_path = task_path + "/../" + cfg_name
    log_dir = os.path.join(task_path, 'runs/rsl_ppo')

    save_items = [cfg_abs_path]
    cfg_saver = ConfigurationSaver(log_dir, save_items, args)

    # config
    cfg = YAML().load(open(cfg_abs_path, 'r'))

    num_envs = cfg['environment']['num_envs']
    process_seeds = np.arange(num_envs) + cfg['environment']['seed'] * num_envs
    assert process_seeds.max() < 2 ** 32

    def make_env(process_idx, test):
        env = gym.make(cfg['environment']['env_name'])
        process_seed = int(process_seeds[process_idx])
        env.seed(process_seed)
        return env

    def make_batch_env(test, n_envs):
        return MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test)
                for idx, env in enumerate(range(n_envs))
            ]
        )

    # batch env for training
    env = make_batch_env(False, num_envs)

    # single env for testing
    test_env = gym.make(cfg['environment']['env_name'])
    test_env.seed(cfg['environment']['seed'])

    if cfg['environment']['record_video']:
        test_env = wrappers.Monitor(test_env, cfg_saver.data_dir, force=True, video_callable=lambda episode: True)

    max_episode_steps = test_env.spec.max_episode_steps
    obs_space = test_env.observation_space
    action_space = test_env.action_space
    total_steps = max_episode_steps * num_envs

    actor_architecture = [64, 64]
    value_net_architecture = [64, 64]

    torch.manual_seed(cfg['environment']['seed'])

    actor_net = nn.Sequential(
                    rslgym_module.EmpiricalNormalization([obs_space.low.size]),
                    rslgym_module.MLP(actor_architecture,
                        nn.LeakyReLU,
                        obs_space.low.size,
                        action_space.low.size)
                    )
    critic_net = nn.Sequential(
                    rslgym_module.EmpiricalNormalization([obs_space.low.size]),
                    rslgym_module.MLP(value_net_architecture,
                        nn.LeakyReLU,
                        obs_space.low.size,
                        1)
                    )

    actor = rslgym_module.Actor(actor_net,
                  rslgym_module.MultivariateGaussianDiagonalCovariance(action_space.low.size, 1.0),
                  obs_space.low.size,
                  action_space.low.size,
                  device)

    critic = rslgym_module.Critic(critic_net, obs_space.low.size, device)

    agent = PPO(actor=actor,
                critic=critic,
                num_envs=num_envs,
                num_transitions_per_env=max_episode_steps,
                num_learning_epochs=cfg['algorithm']['num_epochs'],
                learning_rate=cfg['algorithm']['learning_rate'],
                gamma=cfg['algorithm']['discount_factor'],
                lam=cfg['algorithm']['gae_lam'],
                entropy_coef=cfg['algorithm']['ent_coef'],
                num_mini_batches=cfg['algorithm']['num_mini_batches'],
                device=device,
                log_dir=cfg_saver.data_dir,
                mini_batch_sampling='in_order',
                )

    def obs_to_numpy(obs):
        o = np.array(obs).reshape(len(obs), -1).astype(np.float32)
        return o

    avg_rewards = []
    fig, ax = plt.subplots()
    env.reset()
    obs = env.get_observation()
    obs = obs_to_numpy(obs)
    episode_len = np.zeros(num_envs, dtype="i")

    for update in range(cfg['algorithm']['total_algo_updates']):
        ax.set(xlabel='iteration', ylabel='avg performance', title='average performance')
        ax.grid()
        reward_ll_sum = 0
        done_sum = 0

        # evaluate
        if update % 50 == 0:
            obs_sample = test_env.reset()
            obs_sample = np.array(obs_sample).reshape(1, -1).astype(np.float32)
            for step in range(max_episode_steps):
                action = agent.observe(obs_sample)
                obs_sample, r, dones, _ = test_env.step(action[0])
                obs_sample = np.array(obs_sample).reshape(1, -1).astype(np.float32)
                # reset
                if cfg['environment']['render']:
                    test_env.render()
                if dones:
                    obs_sample = test_env.reset()
                    obs_sample = np.array(obs_sample).reshape(1, -1).astype(np.float32)
                    break

            agent.save_training(cfg_saver.data_dir, update)

        for step in range(cfg['environment']['steps_per_env_and_episode']):
            episode_len += 1
            actor_obs = obs
            critic_obs = obs
            action = agent.observe(actor_obs)
            reward, dones, infos = env.step(action)
            obs = env.get_observation()
            obs = obs_to_numpy(obs)
            reward = np.array(reward)
            dones = np.array(dones)
            resets = episode_len == max_episode_steps
            end = np.logical_or(resets, dones)
            not_end = np.logical_not(end)
            episode_len[end] = 0

            agent.step(value_obs=critic_obs, rews=reward, dones=dones, infos=[])
            done_sum = done_sum + sum(dones)
            reward_ll_sum = reward_ll_sum + sum(reward)
            env.reset(not_end)
            obs = env.get_observation()
            obs = obs_to_numpy(obs)
        agent.update(actor_obs=obs,
                     value_obs=obs,
                     log_this_iteration=update % 1 == 0,
                     update=update)

        average_ll_performance = reward_ll_sum / total_steps
        avg_rewards.append(average_ll_performance)

        actor.distribution.enforce_minimum_std((torch.ones(action_space.low.size)*0.2).to(device))

        if update > 100 and len(avg_rewards) > 100:
            ax.plot(range(len(avg_rewards)), savgol_filter(avg_rewards, 51, 3))
        else:
            ax.plot(range(len(avg_rewards)), avg_rewards)
        fig.savefig(cfg_saver.data_dir + '/demo.png', bbox_inches='tight')

        ax.clear()


if __name__ == "__main__":
    main()
