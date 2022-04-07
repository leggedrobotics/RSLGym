#!/usr/bin/env python3
import os
import argparse
import time
import math

from ruamel.yaml import YAML, dump, RoundTripDumper

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import torch
from torch import nn

from rslgym.wrapper import VecEnvPython
from rslgym_wrapper_cart_pole import cart_pole_example_env
import rslgym.algorithm.modules as rslgym_module
from rslgym.algorithm.agents.ppo import PPO
from rslgym.algorithm.utils import ConfigurationSaver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_name', type=str, default='/cfg.yaml', help='configuration file')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id (-1 for cpu)')
    args = parser.parse_args()
    cfg_name = args.cfg_name

    device = args.gpu if args.gpu > 0 else 'cpu'

    task_path = os.path.dirname(os.path.realpath(__file__))
    rsc_path = task_path + "/../rsc"
    env_path = task_path + "/.."
    cfg_abs_path = task_path + "/../" + cfg_name

    log_dir = os.path.join(task_path, 'runs/rsl_ppo')

    save_items = [env_path+'/Environment.hpp',
                  cfg_abs_path]
    cfg_saver = ConfigurationSaver(log_dir, save_items, args)

    # config
    cfg = YAML().load(open(cfg_abs_path, 'r'))
    impl = cart_pole_example_env(rsc_path, dump(cfg['environment'], Dumper=RoundTripDumper))
    env = VecEnvPython(impl)
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])

    total_steps_per_episode = n_steps * cfg['environment']['num_envs']

    torch.manual_seed(cfg['environment']['seed'])

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    actor_net = rslgym_module.MLP([32, 32], nn.Tanh, obs_size, action_size, 0.5)
    critic_net = rslgym_module.MLP([32, 32], nn.Tanh, obs_size, 1, 0.5)
    actor_dist = rslgym_module.MultivariateGaussianDiagonalCovariance(action_size, 1.0)

    actor = rslgym_module.Actor(actor_net, actor_dist, obs_size, action_size, device)
    critic = rslgym_module.Critic(critic_net, obs_size, device)

    ppo_training = PPO(
        actor=actor,
        critic=critic,
        num_envs=cfg['environment']['num_envs'],
        num_transitions_per_env=n_steps,
        num_learning_epochs=cfg['algorithm']['num_epochs'],
        gamma=cfg['algorithm']['discount_factor'],
        lam=cfg['algorithm']['gae_lam'],
        entropy_coef=cfg['algorithm']['ent_coef'],
        num_mini_batches=cfg['algorithm']['num_mini_batches'],
        device=device,
        log_dir=cfg_saver.data_dir,
        mini_batch_sampling="in_order",
        learning_rate=cfg['algorithm']['learning_rate']
        )

    avg_rewards = []
    fig, ax = plt.subplots()
    for update in range(cfg['algorithm']['total_algo_updates']):

        obs = env.reset()
        reward_ll_sum = 0
        done_sum = 0
        # just keep the number of consecutive up to the latest "done"
        # can be that one env terminates multiple time, count is reset if done is received
        ep_len = np.zeros(shape=env.num_envs)

        if update % 20 == 0:
            env.show_window()
            env.start_recording_video(cfg_saver.data_dir+"/"+str(update)+".mp4")
            for step in range(1*n_steps):
                action_ll, _ = actor.sample(torch.from_numpy(obs).to(ppo_training.device))
                t = time.time()
                obs, reward_ll, dones, _ = env.step(action_ll.cpu().detach().numpy(), True)
                # print(time.time()-t)

            ppo_training.save_training(cfg_saver.data_dir, update)
            obs = env.reset()
            env.stop_recording_video()
            env.hide_window()

        for step in range(n_steps):
            actor_obs = obs
            critic_obs = obs
            action = ppo_training.observe(actor_obs)
            obs, reward, dones, _ = env.step(action, False)
            ep_len[~dones] += 1
            ep_len[dones] = 0
            ppo_training.step(value_obs=critic_obs, rews=reward, dones=dones, infos=[])
            done_sum = done_sum + sum(dones)
            reward_ll_sum = reward_ll_sum + sum(reward)

        ppo_training.update(actor_obs=obs,
                            value_obs=obs,
                            log_this_iteration=update % 1 == 0,
                            update=update)

        average_ll_performance = reward_ll_sum / total_steps_per_episode
        avg_rewards.append(average_ll_performance)


        if update > 100 and len(avg_rewards) > 100:
            ax.plot(range(len(avg_rewards)), savgol_filter(avg_rewards, 51, 3))
        else:
            ax.plot(range(len(avg_rewards)), avg_rewards)
        fig.savefig(cfg_saver.data_dir + '/demo.png', bbox_inches='tight')

        ax.clear()


if __name__ == "__main__":
    main()
