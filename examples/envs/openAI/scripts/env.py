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


from abc import ABCMeta
from abc import abstractmethod


class Env(object, metaclass=ABCMeta):
    """RL learning environment.

    This serves a minimal interface for RL agents.
    """

    @abstractmethod
    def step(self, action):
        raise NotImplementedError()

    @abstractmethod
    def get_observation(self, name=None):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()


class VectorEnv(object, metaclass=ABCMeta):
    """Parallel RL learning environments."""

    @abstractmethod
    def step(self, action):
        raise NotImplementedError()

    @abstractmethod
    def get_observation(self, name=None):
        raise NotImplementedError()

    @abstractmethod
    def reset(self, mask):
        """Reset envs.

        Args:
            mask (Sequence of bool): Mask array that specifies which env to
                skip. If omitted, all the envs are reset.
        """
        raise NotImplementedError()

    @abstractmethod
    def seed(self, seeds):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()

    @property
    def unwrapped(self):
        """Completely unwrap this env.

        Returns:
            VectorEnv: The base non-wrapped VectorEnv instance
        """
        return self
