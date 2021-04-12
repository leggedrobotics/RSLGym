What is the RSLGym?
====================
This is the reinforcement learning framework from the Robotics System Lab (RSL) at ETH Zurich.

It provides an interface to train reinforcement learning agents that are simulated in the RaiSim physics engine.

For efficiency, experience generation in RaiSim is parallelized using a vectorized environment in C++. The vectorized environment is wrapped using pybind11 such that it can be used with RL algorithms implemented in python.

Currently, we provide examples for training agents with a custom PPO implementation and the algorithms provided by PFRL (https://github.com/pfnet/pfrl) which are implemented using pyTorch.

.. toctree::
   :maxdepth: 3
   :caption: Contents

   dependencies
   installation
   examples
   own_env

..
  * :ref:`genindex`
  * :ref:`modindex`
  * :ref:`search`
