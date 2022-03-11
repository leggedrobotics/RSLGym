#!/usr/bin/env python3
from rslgym.wrapper import VecEnvPython  # import python wrapper interface
from rslgym_wrapper_anymal import anymal_example_env
import os
import datetime
import argparse
from ruamel.yaml import YAML, dump, RoundTripDumper
import numpy as np
import torch
import rslgym.algorithm.modules as rslgym_module
from torch import nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument('-w', '--weight_dir', type=str, default='', help='path to trained')
    parser.add_argument('-i', '--iteration', type=int, default=0, help='algo iteration')
    parser.add_argument('-s', '--seconds', type=int, default=10, help='testing duration')
    args = parser.parse_args()
    weight_dir = args.weight_dir
    iteration = args.iteration

    task_path = os.path.dirname(os.path.realpath(__file__))
    rsc_path = task_path + "/../rsc"
    env_path = task_path + "/.."
    save_path = os.path.join(weight_dir, 'testing_' + str(iteration), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(save_path)

    for file in os.listdir(weight_dir):
        if file.startswith('cfg_ppo'):
            cfg_abs_path = weight_dir + '/' + file

    # config
    cfg = YAML().load(open(cfg_abs_path, 'r'))
    cfg['environment']['num_envs'] = 1

    impl = anymal_example_env(rsc_path, dump(cfg['environment'], Dumper=RoundTripDumper))
    env = VecEnvPython(impl)

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    actor_net = nn.Sequential(
                    rslgym_module.EmpiricalNormalization([obs_size]),
                    rslgym_module.MLP([256, 128],
                        nn.Tanh,
                        obs_size,
                        action_size,
                        init_scale=1.4)
                    )

    actor = rslgym_module.Actor(actor_net,
                                rslgym_module.MultivariateGaussianDiagonalCovariance(env.action_space.shape[0], 1.0),
                                env.observation_space.shape[0], env.action_space.shape[0],
                                'cpu')

    snapshot = torch.load(weight_dir + '/snapshot' + str(iteration) + '.pt')
    actor.load_state_dict(snapshot['actor_state_dict'])

    if cfg['environment']['render']:
        env.wrapper.showWindow()

    if cfg['environment']['record_video']:
        env.start_recording_video(save_path + '/test.mp4')

    test_steps = int(args.seconds/cfg['environment']['control_dt'])

    torch.manual_seed(args.seed)

    act = np.ndarray(shape=(1, env.wrapper.getActionDim()), dtype=np.float32)
    _, _, _, new_info = env.step(act, visualize=cfg['environment']['render'])

    ob = env.reset()
    try:
        for i in range(test_steps):
            if i % 100 == 0:
                env.reset()
            act = actor.noiseless_action(torch.from_numpy(ob)).cpu().detach().numpy()
            ob, rew, done, info = env.step(act, visualize=cfg['environment']['render'])


    except KeyboardInterrupt:
        pass

    finally:
        if cfg['environment']['record_video']:
            env.stop_recording_video()


if __name__ == "__main__":
    main()
