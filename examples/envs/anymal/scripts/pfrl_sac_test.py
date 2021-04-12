#!/usr/bin/env python3
import os
import datetime
import argparse
import math
from distutils.version import LooseVersion
from ruamel.yaml import YAML, dump, RoundTripDumper
import numpy as np

import torch
from torch import distributions, nn

import pfrl
from pfrl import replay_buffers, utils
from pfrl.nn.lmbda import Lambda

from rslgym.wrapper import VecEnvPython # import python wrapper interface
from rslgym_wrapper_anymal import anymal_example_env


def main():
    if LooseVersion(torch.__version__) < LooseVersion("1.5.0"):
        raise Exception("This script requires a PyTorch version >= 1.5.0")

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight_dir', type=str, default='', help='path to trained')
    parser.add_argument('-s', '--step_to_load', type=int, default=0, help='step checkpoint to load')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id (-1 for cpu)')
    args = parser.parse_args()
    weight_dir = args.weight_dir
    step_to_load = args.step_to_load

    task_path = os.path.dirname(os.path.realpath(__file__))
    rsc_path = task_path + "/../rsc"
    save_path = os.path.join(weight_dir, 'testing_' + str(step_to_load), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(save_path)

    for file in os.listdir(weight_dir):
        if file.startswith('cfg_sac'):
            cfg_abs_path = weight_dir + '/' + file

    # config
    cfg = YAML().load(open(cfg_abs_path, 'r'))
    cfg['environment']['num_envs'] = 1
    cfg['environment']['num_threads'] = 1
    cfg['environment']['control_dt'] = cfg['testing']['control_dt']
    cfg['environment']['render'] = cfg['testing']['render']

    impl = anymal_example_env(rsc_path, dump(cfg['environment'], Dumper=RoundTripDumper))
    env = VecEnvPython(impl)

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

    agent.load(weight_dir + '/' + str(step_to_load) + '_checkpoint')

    if cfg['testing']['render']:
        env.wrapper.showWindow()

    if cfg['testing']['record_video']:
        env.start_recording_video(save_path + '/test.mp4')

    test_steps = int(cfg['testing']['seconds']/cfg['testing']['control_dt'])

    torch.manual_seed(cfg['environment']['seed'])

    act = np.ndarray(shape=(1, env.wrapper.getActionDim()), dtype=np.float32)
    _, _, _, new_info = env.step(act, visualize=cfg['testing']['render'])

    ob = env.reset()
    try:
        for i in range(test_steps):
            if i % 100 == 0:
                env.reset()
            with agent.eval_mode():
                agent.act_deterministically = True
                act = agent.batch_act(ob)

            ob, rew, done, info = env.step(act, visualize=cfg['testing']['render'])

    except KeyboardInterrupt:
        pass

    finally:
        if cfg['testing']['record_video']:
            env.stop_recording_video()


if __name__ == "__main__":
    main()
