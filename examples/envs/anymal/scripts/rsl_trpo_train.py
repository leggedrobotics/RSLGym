#!/usr/bin/env python3
import os
import argparse
import time
import math

from ruamel.yaml import YAML, dump, RoundTripDumper

import numpy as np

import torch
from torch import nn

import rslgym.algorithm.modules as rslgym_module
from rslgym.algorithm.agents.trpo import TRPO
from rslgym.algorithm.utils import ConfigurationSaver
from rslgym.wrapper import VecEnvPython
from rslgym_wrapper_anymal import anymal_example_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument('--cfg_name', type=str, default='/cfg_trpo.yaml', help='configuration file')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id (-1 for cpu)')
    args = parser.parse_args()
    cfg_name = args.cfg_name

    device = args.gpu if args.gpu > 0 else 'cpu'

    task_path = os.path.dirname(os.path.realpath(__file__))
    rsc_path = task_path + "/../rsc"
    env_path = task_path + "/.."
    cfg_abs_path = task_path + "/.." + cfg_name
    log_dir = os.path.join(task_path, 'runs/rsl_trpo')

    save_items = [env_path+'/Environment.hpp',
                  cfg_abs_path,
                  os.path.realpath(__file__)]

    cfg_saver = ConfigurationSaver(log_dir, save_items, args)

    # config
    cfg = YAML().load(open(cfg_abs_path, 'r'))
    impl = anymal_example_env(rsc_path, dump(cfg['environment'], Dumper=RoundTripDumper))
    env = VecEnvPython(impl)
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])

    total_steps_per_episode = n_steps * cfg['environment']['num_envs']

    torch.manual_seed(args.seed)

    actor_net = rslgym_module.MLP([256, 128],
                                  nn.Tanh,
                                  env.observation_space.shape[0],
                                  env.action_space.shape[0],
                                  init_scale=1.4)

    critic_net = rslgym_module.MLP([256, 128],
                                   nn.Tanh,
                                   env.observation_space.shape[0],
                                   1,
                                   init_scale=1.4)

    actor = rslgym_module.Actor(actor_net,
                  rslgym_module.MultivariateGaussianDiagonalCovariance(env.action_space.shape[0], 1.0),
                  env.observation_space.shape[0], env.action_space.shape[0],
                  device)

    critic = rslgym_module.Critic(critic_net,
                    env.observation_space.shape[0],
                    device)

    agent = TRPO(
        actor=actor,
        critic=critic,
        num_envs=cfg['environment']['num_envs'],
        num_transitions_per_env=n_steps,
        critic_learning_epochs=cfg['algorithm']['critic_learning']['epochs'],
        critic_learning_rate=cfg['algorithm']['critic_learning']['learning_rate'],
        critic_mini_batches=cfg['algorithm']['critic_learning']['num_mini_batches'],
        max_d_kl=cfg['algorithm']['max_kld'],
        gamma=cfg['algorithm']['discount_factor'],
        lam=cfg['algorithm']['gae_lam'],
        entropy_coef=cfg['algorithm']['entropy_coef'],
        device=device,
        log_dir=cfg_saver.data_dir,
        mini_batch_sampling="in_order"
    )

    avg_rewards = []
    for update in range(cfg['algorithm']['total_algorithm_updates']):

        start = time.time()
        obs = env.reset()

        reward_ll_sum = 0
        ep_len = np.zeros(shape=env.num_envs)
        ep_len_collected = []

        if update % cfg['environment']['eval_every_n'] == 0:
            env.show_window()
            if cfg['environment']['record_video']:
                env.start_recording_video(cfg_saver.data_dir + "/" + str(update) + ".mp4")
            for step in range(n_steps):
                action_ll, _ = actor.sample(torch.from_numpy(obs).to(agent.device))
                obs, reward_ll, dones, info = env.step(action_ll.cpu().detach().numpy(), True)

            agent.save_training(cfg_saver.data_dir, update, update)
            obs = env.reset()
            if cfg['environment']['record_video']:
                env.stop_recording_video()
            env.hide_window()

        for step in range(n_steps):
            actor_obs = obs
            critic_obs = obs
            action = agent.observe(actor_obs)
            obs, reward, dones, info = env.step(action, False)
            agent.step(value_obs=critic_obs, rews=reward, dones=dones, infos=[])
            reward_ll_sum = reward_ll_sum + sum(reward)

            ep_len += 1
            if any(dones):
                ep_len_collected += list(ep_len[dones])
                ep_len[dones] = 0
            if step == n_steps - 1:
                for length in list(ep_len):
                    if length == n_steps:
                        ep_len_collected.append(length)

        agent.update(actor_obs=obs,
                     value_obs=obs,
                     log_this_iteration=update % 10 == 0,
                     update=update)
        end = time.time()
        actor.distribution.enforce_minimum_std((torch.ones(12) * 0.2).to(device))

        average_ll_performance = reward_ll_sum / total_steps_per_episode
        avg_rewards.append(average_ll_performance)
        if len(ep_len_collected)> 0:
            avg_ep_leng = sum(ep_len_collected)/len(ep_len_collected) #incorrect
        else:
            avg_ep_leng = n_steps

        agent.writer.add_scalar('Policy/average_reward', average_ll_performance, update)
        agent.writer.add_scalar('Training/elapsed_time_episode', end - start, update)
        agent.writer.add_scalar('Training/fps', total_steps_per_episode / (end - start), update)
        agent.writer.add_scalar('Policy/avg_ep_len', avg_ep_leng, update)

        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(update))
        print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
        print('{:<40} {:>6}'.format("avg_ep_len: ", '{:0.6f}'.format(avg_ep_leng)))
        print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
        print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps_per_episode / (end - start))))
        print('{:<40} {:>6}'.format("std: ", '{}'.format(actor.distribution.log_std.exp())))
        print('----------------------------------------------------\n')


if __name__ == "__main__":
    main()
