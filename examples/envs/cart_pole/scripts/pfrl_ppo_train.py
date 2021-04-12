#!/usr/bin/env python3
import os
import sys
import argparse
import math
from distutils.version import LooseVersion
from ruamel.yaml import YAML, dump, RoundTripDumper
import logging

import torch
from torch import nn

import pfrl
from pfrl import utils

from rslgym.wrapper import VecEnvPython
from rslgym_wrapper_cart_pole import cart_pole_example_env
from rslgym.wrapper import train_agent_batch_with_evaluation_pfrl
from rslgym.wrapper import eval_performance_pfrl
from rslgym.algorithm.utils import ConfigurationSaver


def main():
    if LooseVersion(torch.__version__) < LooseVersion("1.5.0"):
        raise Exception("This script requires a PyTorch version >= 1.5.0")

    # config file arg
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_name', type=str, default='/cfg.yaml', help='configuration file')
    parser.add_argument( "--demo", action="store_true", help="Just run evaluation, not training.")
    parser.add_argument( "--demo-record", action="store_true", help="Save video of demo.")
    parser.add_argument( "--load", type=str, default="", help="Directory to load agent from.")
    parser.add_argument( "--log-interval", type=int, default=1000,
        help="Interval in timesteps between outputting log messages during training",)
    parser.add_argument( "--eval-interval", type=int, default=5000,
        help="Interval in timesteps between evaluations.",)
    parser.add_argument( "--checkpoint-interval", type=int, default=5000,
        help="Interval in timesteps between saving checkpoint",)
    parser.add_argument( "--eval-n-runs", type=int, default=10,
        help="Number of episodes run for each evaluation.",
    )
    parser.add_argument('--gpu', type=int, default=0, help='gpu id (-1 for cpu)')
    args = parser.parse_args()
    cfg_name = args.cfg_name

    # folder config & logdir
    task_path = os.path.dirname(os.path.realpath(__file__))
    rsc_path = task_path + "/../rsc"
    env_path = task_path + "/.."
    cfg_abs_path = task_path + "/../" + cfg_name
    log_dir = os.path.join(task_path, 'runs/pfrl_ppo')

    save_items = [env_path + '/Environment.hpp',
                  cfg_abs_path,
                  __file__]
    cfg_saver = ConfigurationSaver(log_dir, save_items, args)

    # environment
    cfg = YAML().load(open(cfg_abs_path, 'r'))
    impl = cart_pole_example_env(rsc_path, dump(cfg['environment'], Dumper=RoundTripDumper))
    env = VecEnvPython(impl)
    steps_per_episode = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps_per_iteration = steps_per_episode * cfg['environment']['num_envs']
    if total_steps_per_iteration%cfg['algorithm']['num_mini_batches'] > 0.01:
        raise Exception("nminibatches needs to be a multiple of total steps per iteration")

    total_steps_per_minibatch = int(total_steps_per_iteration/cfg['algorithm']['num_mini_batches'])
    log_interval_steps = total_steps_per_iteration  # log (print to terminal) at every algorithm iteration
    eval_interval_steps = total_steps_per_iteration * 20  # evaluate and record video, update tb,
    total_training_steps = cfg['algorithm']['total_algo_updates'] * total_steps_per_iteration
    checkpoint_save_interval_steps = eval_interval_steps

    print(steps_per_episode)
    print('total_steps_per_iteration: ', total_steps_per_iteration)
    print('total_steps_per_minibatch: ', total_steps_per_minibatch)
    print('log_interval_steps: ', log_interval_steps)
    print('eval_interval_steps: ', eval_interval_steps)
    print('total_training_steps: ', total_training_steps)
    print('checkpoint_save_interval_steps: ', checkpoint_save_interval_steps)

    # seeding
    seed = cfg['environment']['seed']
    torch.manual_seed(seed)
    utils.set_random_seed(seed)  # Set a random seed used in PFRL

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
    def ortho_init(layer, gain):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    ortho_init(policy[0], gain=1)
    ortho_init(policy[2], gain=1)
    ortho_init(policy[4], gain=1e-2)
    ortho_init(vf[0], gain=1)
    ortho_init(vf[2], gain=1)
    ortho_init(vf[4], gain=1)

    model = pfrl.nn.Branched(policy, vf)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['algorithm']['learning_rate'], eps=1e-5)

    agent = pfrl.agents.PPO(
        model,
        opt,
        obs_normalizer=None,
        gpu=args.gpu,
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

    # logger settings
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')
    logger = logging.getLogger(__name__)

    if len(args.load) > 0:
        agent.load(args.load)

    if args.demo:
        if cfg['environment']['render']:
            env.show_window()
            if args.demo_record:
                env.start_recording_video(args.load + "/../demo_" + os.path.basename(args.load) + ".mp4")
        eval_stats = eval_performance_pfrl(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
            max_episode_len=steps_per_episode,
            visualize=cfg['environment']['render'],
        )
        if cfg['environment']['render']:
            if args.demo_record:
                env.stop_recording_video()
            env.hide_window()
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        train_agent_batch_with_evaluation_pfrl(
            agent=agent,
            env=env,
            outdir=cfg_saver.data_dir,
            steps=total_training_steps,
            eval_n_steps=steps_per_episode,
            eval_n_episodes=None,  # eval_n_steps or eval_n_episodes, one of them must be none!
            eval_interval=eval_interval_steps,  # in timesteps
            log_interval=log_interval_steps,  # in timesteps
            max_episode_len=steps_per_episode,
            visualize=cfg['environment']['render'],
            use_tensorboard=True,
            checkpoint_freq=checkpoint_save_interval_steps,
            logger=logger
        )


if __name__ == "__main__":
    main()
