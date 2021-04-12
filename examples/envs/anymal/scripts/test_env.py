#!/usr/bin/env python3
import os
import numpy as np
import ruamel.yaml

from rslgym.wrapper import VecEnvPython  # import python wrapper interface
from rslgym_wrapper_anymal import anymal_example_env  # import your environment

task_path = os.path.dirname(os.path.realpath(__file__))
rsc_path = task_path + "/../rsc"
cfg_abs_path = task_path + "/../cfg_ppo.yaml"
cfg = ruamel.yaml.YAML().load(open(cfg_abs_path, 'r'))

dumped_cfg = ruamel.yaml.dump(cfg['environment'], Dumper=ruamel.yaml.RoundTripDumper)
env = VecEnvPython(anymal_example_env(rsc_path, dumped_cfg))

print('action_space ', env.action_space)
print('obs_space ', env.observation_space)
print('num_envs ', env.num_envs)

render = cfg['environment']['render']
if render:
    env.show_window()

obs = env.reset()
info = env.get_info()
# loop for env
for step in range(10000):
    # action = np.zeros((env.num_envs, env.action_space.shape[0])).astype(np.float32)
    action = np.random.randn(env.num_envs, env.action_space.shape[0]).astype(np.float32) * 0.1
    obs, reward, dones, info = env.step(action, visualize=render)
    print('obs ', obs[0])
    print('reward ', reward[0])
    print('dones ', dones[0])
    for key in info.keys():
        print(key, info[key][0])

    # you can pass numpy array with name.
    pass_info = {'test_info': np.arange(env.num_envs * 5, dtype=np.float32).reshape(env.num_envs, -1),
                 'curriculum': np.array([[0.2]], dtype=np.float32).repeat(env.num_envs, axis=0)}

    # you can add any key at any timing.
    if step % 10 == 0:
        pass_info['trigger'] = np.array([[0.0]], dtype=np.float32).repeat(env.num_envs, axis=0)

    # send info to env
    env.set_info(pass_info)

if render:
    env.hide_window()
