#!/usr/bin/env python3
from rslgym.wrapper import VecEnvPython # import python wrapper interface
from rslgym_wrapper_cart_pole import cart_pole_example_env
import os
import datetime
import argparse
from ruamel.yaml import YAML, dump, RoundTripDumper
import numpy as np
import torch
import rslgym.algorithm.modules as rslgym_module
from torch import nn
import matplotlib.pyplot as plt


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
        if file.startswith('cfg'):
            cfg_abs_path = weight_dir + '/' + file

    # config
    cfg = YAML().load(open(cfg_abs_path, 'r'))
    cfg['environment']['num_envs'] = 1

    impl = cart_pole_example_env(rsc_path, dump(cfg['environment'], Dumper=RoundTripDumper))
    env = VecEnvPython(impl)

    actor_net = rslgym_module.MLP([32, 32],
                               nn.Tanh,
                               env.observation_space.shape[0],
                               env.action_space.shape[0])

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

    # containers for analysis
    actions = np.zeros(shape=(2, test_steps), dtype=np.float32)
    obs = np.zeros(shape=(4, test_steps), dtype=np.float32)

    ob = env.reset()
    try:
        for i in range(test_steps):
            if i % 100 == 0:
                env.reset()
            act = actor.noiseless_action(torch.from_numpy(ob)).cpu().detach().numpy()
            ob, rew, done, info = env.step(act, visualize=cfg['environment']['render'])
            obs[:, i] = ob
            actions[0, i] = info['action']
            actions[1, i] = act

    except KeyboardInterrupt:
        pass

    finally:
        if cfg['environment']['record_video']:
            env.stop_recording_video()

        if cfg['environment']['render']:
            env.wrapper.hideWindow()

        import matplotlib
        matplotlib.use('TKAgg')
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(actions[0, :], label='applied action')
        plt.plot(actions[1, :], label='nn action')
        plt.grid()
        plt.legend()

        plt.figure()
        plt.plot(obs[0, :], label='cart pos')
        plt.plot(obs[2, :], label='cart vel')
        plt.grid()
        plt.legend()

        plt.figure()
        plt.plot(obs[1, :], label='pend pos')
        plt.plot(obs[3, :], label='pend vel')
        plt.grid()
        plt.legend()

        plt.show(block=False)
        input('press [ENTER] to exit')

if __name__ == "__main__":
    main()
