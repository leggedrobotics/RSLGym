import os
import numpy as np
from gym import spaces
import warnings
from .base_vec_env import VecEnv


class RaisimGymVecEnv(VecEnv):

    def __init__(self, impl, activation_path=os.environ['HOME'] + '/.raisim/activation.raisim'):
        self.wrapper = impl
        self.wrapper.setActivationKey(activation_path)
        self.wrapper.init()
        self.num_obs = self.wrapper.getObDim()
        self.num_acts = self.wrapper.getActionDim()
        self._observation_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf,
                                             dtype=np.float32)
        self._action_space = spaces.Box(np.ones(self.num_acts) * -1., np.ones(self.num_acts) * 1., dtype=np.float32)
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros((self.num_envs), dtype=np.bool)
        self._infoNames = self.wrapper.getInfoNames()
        self._infoDims = self.wrapper.getInfoDims()
        self._info = {self._infoNames[i]: np.zeros([self.num_envs, self._infoDims[i]], dtype=np.float32)
                      for i in range(len(self._infoNames))}
        self.rewards = [[] for _ in range(self.num_envs)]

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def step(self, action, visualize=False):
        assert action.shape[0] == self.num_envs and action.shape[1] == self.num_acts
        action = action.astype(np.float32)
        if not visualize:
            self.wrapper.step(action, self._observation, self._reward, self._done)
        else:
            self.wrapper.testStep(action, self._observation, self._reward, self._done)

        self._info = self.wrapper.getInfo()

        return self._observation.copy(), self._reward.copy(), self._done.copy(), self._info.copy()

    def reset(self, mask=None):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset(self._observation)
        return self._observation.copy()

    def get_info(self):
        self._info = self.wrapper.getInfo()
        return self._info

    def set_info(self, info):
        assert isinstance(info, dict)
        info = dict([key, value.astype(np.float32).reshape(self.num_envs, -1)] for key, value in info.items())
        self.wrapper.setInfo(info.copy())

    def reset_and_update_info(self):
        return self.reset(), self._update_epi_info()

    def _update_epi_info(self):
        info = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            eprew = sum(self.rewards[i])
            eplen = len(self.rewards[i])
            epinfo = {"r": eprew, "l": eplen}
            info[i]['episode'] = epinfo
            self.rewards[i].clear()

        return info

    def render(self, mode='human'):
        raise RuntimeError('This method is not implemented')

    def close(self):
        self.wrapper.close()

    def start_recording_video(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_recording_video(self):
        self.wrapper.stopRecordingVideo()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def step_async(self):
        raise RuntimeError('This method is not implemented')

    def step_wait(self):
        raise RuntimeError('This method is not implemented')

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.

        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        raise RuntimeError('This method is not implemented')

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.

        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        raise RuntimeError('This method is not implemented')

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.

        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        raise RuntimeError('This method is not implemented')

    def show_window(self):
        self.wrapper.showWindow()

    def hide_window(self):
        self.wrapper.hideWindow()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def info_names(self):
        return self._infoNames
