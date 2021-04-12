from datetime import datetime
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from rslgym.algorithm.storage.rollout import RolloutStorage


class TRPO:
    def __init__(self,
                 actor,
                 critic,
                 num_envs,
                 num_transitions_per_env,
                 critic_learning_epochs,
                 critic_mini_batches,
                 critic_learning_rate=5e-4,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 entropy_coef=0.0,
                 max_d_kl=0.01,
                 max_line_search_iteration=5,
                 conjugate_gradient_damping=1e-2,
                 log_dir='run',
                 device='cpu',
                 mini_batch_sampling='shuffle',
                 log_intervals=10):

        # PPO components
        self.actor = actor
        self.critic = critic
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor.obs_shape, critic.obs_shape,
                                      actor.action_shape, device)

        if mini_batch_sampling is 'shuffle':
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        elif mini_batch_sampling is 'in_order':
            self.batch_sampler = self.storage.mini_batch_generator_inorder
        else:
            raise NameError(
                mini_batch_sampling + ' is not a valid sampling method. Use one of the followings: shuffle, order')

        self.critic_optimizer = optim.Adam([*self.critic.parameters()], lr=critic_learning_rate)
        self.device = device

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # TRPO parameters
        self.clip_param = clip_param
        self.critic_learning_epochs = critic_learning_epochs
        self.critic_mini_batches = critic_mini_batches
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_d_kl = max_d_kl
        self.max_line_search_iteration = max_line_search_iteration
        self.conjugate_gradient_damping = conjugate_gradient_damping

        # Log
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.ep_infos = []
        self.log_intervals = log_intervals

        self.update_num = 0

        # temps
        self.actions = None
        self.actions_log_prob = None
        self.actor_obs = None

    def observe(self, actor_obs):
        self.actor_obs = actor_obs.copy()
        with torch.no_grad():
            self.actions, self.actions_log_prob = self.actor.sample(torch.from_numpy(self.actor_obs).to(self.device))
        # self.actions = np.clip(self.actions.numpy(), self.env.action_space.low, self.env.action_space.high)
        return self.actions.cpu().numpy()

    def step(self, value_obs, rews, dones, infos):
        values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))
        self.storage.add_transitions(self.actor_obs, value_obs, self.actions, rews, dones, values,
                                     self.actions_log_prob)

        # Book keeping
        for info in infos:
            ep_info = info.get('episode')
            if ep_info is not None:
                self.ep_infos.append(ep_info)

    def update(self, actor_obs, value_obs, log_this_iteration, update):
        with torch.no_grad():
            last_values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))
        # dones = torch.from_numpy(dones).to(self.device).reshape([-1, 1])

        # Learning step
        self.storage.compute_returns(last_values.to(self.device), self.gamma, self.lam)
        mean_value_loss, surrogate_loss, infos = self._train_step()
        self.storage.clear()

        self.update_num = update

        self.ep_infos.clear()

        mean_std = self.actor.distribution.log_std.exp().mean()

        self.writer.add_scalar('Loss/value_function', mean_value_loss, update)
        self.writer.add_scalar('Loss/surrogate', surrogate_loss, update)
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), update)

    def _train_step(self):
        mean_value_loss = 0
        surrogate_loss = 0
        # Value function update
        for epoch in range(self.critic_learning_epochs):
            for actor_obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
                    in self.storage.mini_batch_generator_inorder(self.critic_mini_batches):

                # update critic
                value_batch = self.critic.evaluate(critic_obs_batch)
                value_loss = (returns_batch - value_batch).pow(2).mean()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()
                mean_value_loss += value_loss.item()

        # Policy update
        actor_obs_batch = self.storage.actor_obs.view(-1, *self.storage.actor_obs.size()[2:])
        critic_obs_batch = self.storage.critic_obs.view(-1, *self.storage.critic_obs.size()[2:])
        actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))
        target_values_batch = self.storage.values.view(-1, 1)
        advantages_batch = self.storage.advantages.view(-1, 1)
        returns_batch = self.storage.returns.view(-1, 1)
        old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)

        actions_log_prob_batch, entropy_batch = self.actor.evaluate(actor_obs_batch, actions_batch)
        old_actions_log_prob_batch = torch.squeeze(old_actions_log_prob_batch)
        advantages_batch = torch.squeeze(advantages_batch)

        surrogate_loss = self._surrogate(actions_log_prob_batch, old_actions_log_prob_batch, entropy_batch, advantages_batch)
        KL = self._kl_div_log(old_actions_log_prob_batch, actions_log_prob_batch)

        parameters = list(self.actor.parameters())
        g = self._flat_grad(surrogate_loss, parameters, retain_graph=True)
        d_kl = self._flat_grad(KL, parameters, create_graph=True)
        # Create graph, because we will call backward() on it (for HVP)

        def hessian_vector_product(v):
            return self._flat_grad(d_kl @ v, parameters, retain_graph=True)

        def fisher_vector_product_func(vec):
            fvp = hessian_vector_product(vec)
            return fvp + self.conjugate_gradient_damping * vec

        search_dir = self._conjugate_gradient(fisher_vector_product_func, g)

        # criterion for line search
        def criterion(step):
            self._apply_update(step)
            with torch.no_grad():
                new_actions_log_prob_batch, new_entropy_batch = self.actor.evaluate(actor_obs_batch,
                                                                                    actions_batch)
                L_new = self._surrogate(new_actions_log_prob_batch, old_actions_log_prob_batch,
                                        new_entropy_batch, advantages_batch)
                KL_new = self._kl_div_log(old_actions_log_prob_batch, new_actions_log_prob_batch)

            L_improvement = L_new - surrogate_loss

            if L_improvement > 0 and KL_new <= self.max_d_kl:
                return True

            # revert
            self._apply_update(-step)
            return False

        # Line search and update parameter
        dId = float(search_dir.dot(fisher_vector_product_func(search_dir)))

        scale = (2.0 * self.max_d_kl / (dId + 1e-8)) ** 0.5

        max_step = scale * search_dir
        i = 0
        if self.max_line_search_iteration == 0:
            self._apply_update(max_step)
        else:
            for i in range(self.max_line_search_iteration + 1):
                if criterion((0.6 ** i) * max_step):
                    break

        num_updates = self.critic_learning_epochs * self.critic_mini_batches
        mean_value_loss /= num_updates

        return mean_value_loss, surrogate_loss, locals()

    def _surrogate(self, log_prob, log_prob_old, entropy, advs):
        """Compute a gain to maximize."""
        prob_ratio = torch.exp(log_prob - log_prob_old)
        mean_entropy = torch.mean(entropy)
        surrogate_gain = torch.mean(prob_ratio * advs)
        return surrogate_gain + self.entropy_coef * mean_entropy

    def _kl_div(self, p, q):
        p = p.detach()
        return (p * (p.log() - q.log())).sum(-1).mean()

    def _kl_div_log(self, p_log, q_log):
        p_log = p_log.detach()
        return (torch.exp(p_log) * (p_log - q_log)).sum(-1).mean()

    def _flat_grad(self, y, x, retain_graph=False, create_graph=False):
        if create_graph:
            retain_graph = True

        g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        g = torch.cat([t.view(-1) for t in g])
        return g

    def _conjugate_gradient(self, A_product_func, b, tol=1e-10, max_iter=10):
        """Conjugate Gradient (CG) method.
        This function solves Ax=b for the vector x, where A is a real
        positive-definite matrix and b is a real vector.
        Args:
            A_product_func (callable): Callable that returns the product of the
                matrix A and a given vector.
            b (numpy.ndarray or cupy.ndarray): The vector b.
            tol (float): Tolerance parameter for early stopping.
            max_iter (int): Maximum number of iterations.
        Returns:
            numpy.ndarray or cupy.ndarray: The solution.
                The array module will be the same as the argument b's.
        """
        x = torch.zeros_like(b)
        r0 = b - A_product_func(x)
        p = r0
        for _ in range(max_iter):
            a = torch.matmul(r0, r0) / torch.matmul(A_product_func(p), p)
            x = x + p * a
            r1 = r0 - A_product_func(p) * a
            if torch.norm(r1) < tol:
                return x
            b = torch.matmul(r1, r1) / torch.matmul(r0, r0)
            p = r1 + b * p
            r0 = r1
        return x

    def _apply_update(self, grad_flattened):
        n = 0
        for p in self.actor.parameters():
            numel = p.numel()
            g = grad_flattened[n:n + numel].view(p.shape)
            p.data += g
            n += numel

    def save_training(self, dirname, curriculum, update_num):
        print('save training. current curriculum is ', curriculum)
        path = dirname + "/snapshot" + str(update_num) + ".pt"
        torch.save({
            'epoch': self.update_num,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.critic_optimizer.state_dict(),
            'curriculum': curriculum,
        }, path)
        print('saved training to ', path)

    def load_training(self, dirname, update):
        path = dirname + "/snapshot" + str(update) + ".pt"
        print('load training from ', path)
        snapshot = torch.load(path)
        self.actor.load_state_dict(snapshot['actor_state_dict'])
        self.critic.load_state_dict(snapshot['critic_state_dict'])
        self.critic_optimizer.load_state_dict(snapshot['optimizer_state_dict'])
        print('loaded epoch is {}. curriculum is {}'.format(snapshot['epoch'], snapshot['curriculum']))
        return snapshot['epoch'], snapshot['curriculum']
