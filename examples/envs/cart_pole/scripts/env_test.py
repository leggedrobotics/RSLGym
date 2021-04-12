#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper
from rslgym.wrapper import VecEnvPython as vecEnv
from rslgym_wrapper_cart_pole import cart_pole_example_env
import os
import numpy as np

# configuration
task_path = os.path.dirname(os.path.realpath(__file__))
rsc_path = task_path + "/../rsc"
cfg_path = task_path + '/../cfg.yaml'
cfg = YAML().load(open(cfg_path, 'r'))

print("Loaded cfg from {}\n".format(cfg_path))

cfg['environment']['num_envs'] = 1
cfg['environment']['render'] = True
env = vecEnv(cart_pole_example_env(rsc_path, dump(cfg['environment'], Dumper=RoundTripDumper)))

env.wrapper.showWindow()

ob = env.reset()

# env.start_recording_video("test/env_test.mp4")

for i in range(1000):
    if i % 1 == 0:
        env.reset()

    act = np.ndarray(shape=(1, 1), dtype=np.float32)
    act[:] = 0
    ob, rew, done, newInfo = env.step(act, visualize=True)
    print(ob[0, 1])


# env.stop_recording_video()
