#!/usr/bin/env python3
import os
import sys
import argparse
import math
from distutils.version import LooseVersion
from ruamel.yaml import YAML, dump, RoundTripDumper
import logging
import numpy as np

import torch
from torch import distributions, nn

import pfrl
from pfrl import replay_buffers, utils
from pfrl.nn.lmbda import Lambda

from rslgym.wrapper import VecEnvPython  # import python wrapper interface
from rslgym.wrapper import train_agent_batch_with_evaluation_pfrl
from rslgym.wrapper import eval_performance_pfrl
from rslgym_wrapper_anymal import anymal_example_env
from rslgym.algorithm.utils import ConfigurationSaver

def main():
    if LooseVersion(torch.__version__) < LooseVersion("1.5.0"):
        raise Exception("This script requires a PyTorch version >= 1.5.0")

    # config file arg
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_name', type=str, default='cfg_sac.yaml', help='configuration file')
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
    log_dir = os.path.join(task_path, 'runs/pfrl_sac')

    save_items = [env_path + '/Environment.hpp',
                  cfg_abs_path,
                  __file__]
    if not args.demo:
        cfg_saver = ConfigurationSaver(log_dir, save_items, args)


    # environment
    cfg = YAML().load(open(cfg_abs_path, 'r'))
    impl = anymal_example_env(rsc_path, dump(cfg['environment'], Dumper=RoundTripDumper))
    env = VecEnvPython(impl)
    steps_per_episode = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps_per_iteration = steps_per_episode * cfg['environment']['num_envs']

    total_training_steps = cfg['algorithm']['total_algorithm_updates'] * total_steps_per_iteration

    obs_space = env.observation_space
    action_space = env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    # seeding
    seed = cfg['environment']['seed']
    torch.manual_seed(seed)
    utils.set_random_seed(seed)  # Set a random seed used in PFRL

    obs_size = obs_space.low.size
    action_size = action_space.low.size

    def squashed_diagonal_gaussian_head(x):
        assert x.shape[-1] == action_size * 2
        mean, log_scale = torch.chunk(x, 2, dim=1)
        log_scale = torch.clamp(log_scale, -20.0, 2.0)
        var = torch.exp(log_scale * 2)
        base_distribution = distributions.Independent(
            distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
        )
        # cache_size=1 is required for numerical stability
        return distributions.transformed_distribution.TransformedDistribution(
            base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
        )

    policy = nn.Sequential(
        nn.Linear(obs_size, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, action_size * 2),
        Lambda(squashed_diagonal_gaussian_head),
    )
    torch.nn.init.xavier_uniform_(policy[0].weight)
    torch.nn.init.xavier_uniform_(policy[2].weight)
    torch.nn.init.xavier_uniform_(policy[4].weight, gain=1.0)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg['algorithm']['learning_rate'])

    def make_q_func_with_optimizer():
        q_func = nn.Sequential(
            pfrl.nn.ConcatObsAndAction(),
            nn.Linear(obs_size + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        torch.nn.init.xavier_uniform_(q_func[1].weight)
        torch.nn.init.xavier_uniform_(q_func[3].weight)
        torch.nn.init.xavier_uniform_(q_func[5].weight)
        q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=cfg['algorithm']['learning_rate'])
        return q_func, q_func_optimizer

    q_func1, q_func1_optimizer = make_q_func_with_optimizer()
    q_func2, q_func2_optimizer = make_q_func_with_optimizer()

    rbuf = replay_buffers.ReplayBuffer(cfg['algorithm']['replay_buffer_size'])

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = pfrl.agents.SoftActorCritic(
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        rbuf,
        gamma=cfg['algorithm']['discount_factor'],
        replay_start_size=cfg['algorithm']['replay_start_size'],
        gpu=args.gpu,
        minibatch_size=cfg['algorithm']['minibatch_size'],
        burnin_action_func=burnin_action_func,
        entropy_target=-action_size,
        temperature_optimizer_lr=cfg['algorithm']['temperature_optimizer_lr'],
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
            eval_n_episodes=None,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            max_episode_len=steps_per_episode,
            visualize=cfg['environment']['render'],
            use_tensorboard=True,
            checkpoint_freq=args.checkpoint_interval,
            logger=logger
        )


if __name__ == "__main__":
    main()
