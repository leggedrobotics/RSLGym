# MIT License
#
# Copyright (c) 2020 Preferred Networks, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values.
    Args:
        shape (int or tuple of int): Shape of input values except batch axis.
        batch_axis (int): Batch axis.
        eps (float): Small value for stability.
        dtype (dtype): Dtype of input values.
        until (int or None): If this arg is specified, the link learns input
            values until the sum of batch sizes exceeds it.
    """

    def __init__(
        self,
        shape,
        batch_axis=0,
        eps=1e-2,
        dtype=np.float32,
        until=None,
        clip_threshold=None,
    ):
        super(EmpiricalNormalization, self).__init__()
        dtype = np.dtype(dtype)
        self.batch_axis = batch_axis
        self.eps = dtype.type(eps)
        self.until = until
        self.clip_threshold = clip_threshold
        self.register_buffer(
            "_mean",
            torch.tensor(np.expand_dims(np.zeros(shape, dtype=dtype), batch_axis)),
        )
        self.register_buffer(
            "_var",
            torch.tensor(np.expand_dims(np.ones(shape, dtype=dtype), batch_axis)),
        )
        self.register_buffer("count", torch.tensor(0))

        # cache
        self._cached_std_inverse = None

    @property
    def mean(self):
        return torch.squeeze(self._mean, self.batch_axis).clone()

    @property
    def std(self):
        return torch.sqrt(torch.squeeze(self._var, self.batch_axis)).clone()

    @property
    def _std_inverse(self):
        if self._cached_std_inverse is None:
            self._cached_std_inverse = (self._var + self.eps) ** -0.5

        return self._cached_std_inverse

    def experience(self, x):
        """Learn input values without computing the output values of them"""

        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[self.batch_axis]
        if count_x == 0:
            return

        self.count += count_x
        rate = count_x / self.count.float()
        assert rate > 0
        assert rate <= 1

        var_x, mean_x = torch.var_mean(
            x, axis=self.batch_axis, keepdims=True, unbiased=False
        )
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))

        # clear cache
        self._cached_std_inverse = None

    def forward(self, x, update=True):
        """Normalize mean and variance of values based on emprical values.
        Args:
            x (ndarray or Variable): Input values
            update (bool): Flag to learn the input values
        Returns:
            ndarray or Variable: Normalized output values
        """

        if update:
            self.experience(x)

        if not x.is_cuda:
            self._cached_std_inverse = None
        normalized = (x - self._mean) * self._std_inverse
        if self.clip_threshold is not None:
            normalized = torch.clamp(
                normalized, -self.clip_threshold, self.clip_threshold
            )
        if not x.is_cuda:
            self._cached_std_inverse = None
        return normalized

    def inverse(self, y):
        std = torch.sqrt(self._var + self.eps)
        return y * std + self._mean

    def load_numpy(self, mean, var, count, device='cpu'):
        self._mean = torch.from_numpy(np.expand_dims(mean, self.batch_axis)).to(device)
        self._var = torch.from_numpy(np.expand_dims(var, self.batch_axis)).to(device)
        self.count = torch.tensor(count).to(device)


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, arr):
        if np.isnan(arr).any():
            print('array contains nan value')
            print('array size is', arr.shape)
            print('nan index', list(map(tuple, np.where(np.isnan(arr[0])))))
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        if self.count < 100000000:
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, obs, clip):
        normalized_obs = np.clip((obs - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)
        return normalized_obs

    def save(self, dir_name, iteration, prefix=""):
        mean_file_name = dir_name + "/mean" + prefix + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + prefix + str(iteration) + ".csv"
        count_file_name = dir_name + "/count" + prefix + str(iteration) + ".txt"
        np.savetxt(mean_file_name, self.mean)
        np.savetxt(var_file_name, self.var)
        np.savetxt(count_file_name, np.array([self.count]))

    def load(self, dir_name, iteration, prefix=""):
        mean_file_name = dir_name + "/mean" + prefix + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + prefix + str(iteration) + ".csv"
        count_file_name = dir_name + "/count" + prefix + str(iteration) + ".txt"
        self.mean = np.loadtxt(mean_file_name, dtype=np.float32)
        self.var = np.loadtxt(var_file_name, dtype=np.float32)
        self.count = np.loadtxt(count_file_name, dtype=np.float32)

