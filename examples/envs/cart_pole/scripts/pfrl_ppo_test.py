#!/usr/bin/env python3
from rslgym.wrapper import VecEnvPython  # import python wrapper interface
from rslgym_wrapper_cart_pole import cart_pole_example_env
import os
import datetime
import argparse
from ruamel.yaml import YAML, dump, RoundTripDumper
import numpy as np
import math
import torch
from torch import nn
from distutils.version import LooseVersion

import pfrl

import matplotlib.pyplot as plt

def main():
    if LooseVersion(torch.__version__) < LooseVersion("1.5.0"):
        raise Exception("This script requires a PyTorch version >= 1.5.0")

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight_dir', type=str, default='', help='path to trained')
    parser.add_argument('-s', '--step_to_load', type=int, default=0, help='step checkpoint to load')
    parser.add_argument('-t', '--test_steps', type=int, default=10, help='testing duration in secs')
    args = parser.parse_args()
    weight_dir = args.weight_dir
    step_to_load = args.step_to_load

    task_path = os.path.dirname(os.path.realpath(__file__))
    rsc_path = task_path + "/../rsc"
    save_path = os.path.join(weight_dir, 'testing_' + str(step_to_load), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(save_path)

    for file in os.listdir(weight_dir):
        if file.startswith('cfg'):
            cfg_abs_path = weight_dir + '/' + file

    # config
    cfg = YAML().load(open(cfg_abs_path, 'r'))
    cfg['environment']['num_envs'] = 1
    cfg['environment']['num_threads'] = 1

    impl = cart_pole_example_env(rsc_path, dump(cfg['environment'], Dumper=RoundTripDumper))
    env = VecEnvPython(impl)

    steps_per_episode = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps_per_iteration = steps_per_episode * cfg['environment']['num_envs']
    if total_steps_per_iteration%cfg['algorithm']['num_mini_batches'] > 0.01:
        raise Exception("nminibatches needs to be a multiple of total steps per iteration")
    total_steps_per_minibatch = int(total_steps_per_iteration/cfg['algorithm']['num_mini_batches'])

    # actor & critic
    policy = torch.nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, env.action_space.shape[0]),
        pfrl.policies.GaussianHeadWithStateIndependentCovariance(
            action_size=env.action_space.shape[0],
            var_type="diagonal",
            var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        ),
    )

    vf = torch.nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
    )

    model = pfrl.nn.Branched(policy, vf)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['algorithm']['learning_rate'], eps=1e-5)

    agent = pfrl.agents.PPO(
        model,
        opt,
        obs_normalizer=None,
        gpu=-1,
        value_func_coef=cfg['algorithm']['vf_coef'],
        update_interval=total_steps_per_iteration,
        minibatch_size=total_steps_per_minibatch,
        epochs=cfg['algorithm']['num_epochs'],
        clip_eps_vf=None,
        entropy_coef=cfg['algorithm']['ent_coef'],
        standardize_advantages=True,
        gamma=cfg['algorithm']['discount_factor'],
        lambd=cfg['algorithm']['gae_lam']
    )

    agent.load(weight_dir + '/' + str(step_to_load) + '_checkpoint')

    if cfg['environment']['render']:
        env.wrapper.showWindow()

    if cfg['environment']['record_video']:
        env.start_recording_video(save_path + '/test.mp4')

    test_steps = int(args.test_steps/cfg['environment']['control_dt'])

    torch.manual_seed(cfg['environment']['seed'])

    act = np.ndarray(shape=(1, env.wrapper.getActionDim()), dtype=np.float32)
    _, _, _, new_info = env.step(act, visualize=cfg['environment']['render'])

    ob = env.reset()
    try:
        for i in range(test_steps):
            if i % 100 == 0:
                env.reset()
            with agent.eval_mode():
                agent.act_deterministically = True
                act = agent.batch_act(ob)

            ob, rew, done, info = env.step(act, visualize=cfg['environment']['render'])

    except KeyboardInterrupt:
        pass

    finally:
        if cfg['environment']['record_video']:
            env.stop_recording_video()


if __name__ == "__main__":
    main()
