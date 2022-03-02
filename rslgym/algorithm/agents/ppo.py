from datetime import datetime
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from rslgym.algorithm.storage.rollout import RolloutStorage
from dataclasses import dataclass


@dataclass
class PPOLog:
    writer: SummaryWriter
    iteration: int = 0

    @dataclass
    class PrintLog:
        steps: int = 0
        average_reward: float = 0.
        average_dones: float = 0.
        wall_time: float = time.time()
        noise_std: np.array = np.zeros(1)

        reward_sum: float = 0.
        dones_sum: float = 0.
    print_log: PrintLog = PrintLog()

    @dataclass
    class SummaryLog:
        @dataclass
        class Loss:
            value_loss: float = 0.
            surrogate_loss: float = 0.
        loss: Loss = Loss()

        @dataclass
        class Policy:
            average_reward: float = 0.
            average_dones: float = 0.
            mean_noise_std: float = 0.
            discounted_rewards: float = 0.
        policy: Policy = Policy()

    summary_log: SummaryLog = SummaryLog()

    def update(self, reward, dones):
        self.print_log.steps += reward.shape[0]
        self.print_log.reward_sum += np.sum(reward)
        self.print_log.dones_sum += np.sum(dones)
        self.print_log.average_reward = self.print_log.reward_sum / self.print_log.steps
        self.print_log.average_dones = self.print_log.dones_sum / self.print_log.steps
        self.summary_log.policy.average_reward = self.print_log.average_reward
        self.summary_log.policy.average_dones = self.print_log.average_dones

    def reset(self):
        self.print_log.steps = 0
        self.print_log.reward_sum = 0.
        self.print_log.dones_sum = 0.
        self.print_log.wall_time = time.time()

    def __str__(self):
        s = ('----------------------------------------------------\n'
            +'{:>6}th iteration\n'.format(self.iteration)
            +'{:<40} {:>6}\n'.format("average reward: ", '{:0.10f}'.format(self.print_log.average_reward))
            +'{:<40} {:>6}\n'.format("dones: ", '{:0.6f}'.format(self.print_log.average_dones))
            +'{:<40} {:>6}\n'.format("time elapsed in this iteration: ", '{:6.4f}'.format(time.time() - self.print_log.wall_time))
            +'{:<40} {:>6}\n'.format("fps: ", '{:6.0f}'.format(self.print_log.steps/ (time.time() - self.print_log.wall_time + 1e-8)))
            +'{:<40} \n{}\n'.format("noise_std: ", self.print_log.noise_std))
        return s

    def write_summary(self, extra_info={}):
        for k, v in self.summary_log.__dict__.items():
            for k_, v_ in v.__dict__.items():
                name = "{}/{}".format(k, k_)
                self.writer.add_scalar(name, v_, self.iteration)
        for k, v in extra_info.items():
            name = "Extra/{}".format(k)
            self.writer.add_scalar(name, v, self.iteration)


class PPO:
    def __init__(self,
                 actor,
                 critic,
                 num_envs,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=0.5,
                 entropy_coef=0.0,
                 learning_rate=5e-4,
                 learning_rate_gamma=1.0,  # by default does not decay the learning rate.
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 log_dir='run',
                 device='cpu',
                 mini_batch_sampling='shuffle',
                 log_intervals=10):

        # PPO components
        self.actor = actor
        self.critic = critic
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor.obs_shape, critic.obs_shape, actor.action_shape, device)

        if mini_batch_sampling is 'shuffle':
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        elif mini_batch_sampling is 'in_order':
            self.batch_sampler = self.storage.mini_batch_generator_inorder
        else:
            raise NameError(mini_batch_sampling + ' is not a valid sampling method. Use one of the followings: shuffle, order')

        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=learning_rate)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, learning_rate_gamma, last_epoch=-1, verbose=True)
        self.device = device

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Log
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.log = PPOLog(writer=self.writer)
        self.ep_infos = {}
        self.log_intervals = log_intervals
        self.update_num = 0

        # temps
        self.actions = None
        self.actions_log_prob = None
        self.actor_obs = None

    def observe(self, actor_obs):
        with torch.no_grad():
            self.actor_obs = actor_obs
            self.actions, self.actions_log_prob = self.actor.sample(torch.from_numpy(actor_obs).to(self.device))
        return self.actions.cpu().numpy()

    def step(self, value_obs, rews, dones, infos):
        with torch.no_grad():
            values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))
        self.storage.add_transitions(self.actor_obs, value_obs, self.actions, rews, dones, values,
                                     self.actions_log_prob)

        self.log.update(rews, dones)
        # Book keeping
        for info in infos:
            ep_info = info.get('episode')
            if ep_info is not None:
                self.ep_infos.update(ep_info)

    def act(self, actor_obs, noiseless=True):
        with torch.no_grad():
            if noiseless:
                actions = self.actor.noiseless_action(torch.from_numpy(actor_obs).to(self.device))
            else:
                actions, _ = self.actor.sample(torch.from_numpy(actor_obs).to(self.device))
        return actions.cpu().numpy()

    def update(self, actor_obs, value_obs, log_this_iteration=False, update=0):
        last_values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))

        # Learning step
        self.storage.compute_returns(last_values.to(self.device), self.gamma, self.lam)
        infos = self._train_step()
        self.storage.clear()
        self.update_num = update

        if log_this_iteration:
            self.log.iteration = update
            self.add_log({**locals(), **infos, 'ep_infos': self.ep_infos, 'it': update})

        self.ep_infos.clear()

    def add_log(self, extra_info):        
        self.log.summary_log.policy.discounted_rewards = self.storage.returns.mean().item()
        self.log.summary_log.policy.mean_noise_std = self.actor.distribution.log_std.exp().mean().item()
        self.log.print_log.noise_std = self.actor.distribution.log_std.exp().detach().cpu().numpy()

        self.log.write_summary(extra_info=extra_info['ep_infos'])
        print(self.log)
        self.log.reset()

    def _train_step(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        for epoch in range(self.num_learning_epochs):
            for actor_obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
                    in self.storage.mini_batch_generator_inorder(self.num_mini_batches):

                actions_log_prob_batch, entropy_batch = self.actor.evaluate(actor_obs_batch, actions_batch)
                value_batch = self.critic.evaluate(critic_obs_batch)

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        self.lr_scheduler.step()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        self.log.summary_log.loss.value_loss = mean_value_loss
        self.log.summary_log.loss.surrogate_loss = mean_surrogate_loss

        return locals()

    def save_training(self, dirname, epoch=0, info={}):
        print("save training.")
        path = dirname + "/snapshot" + str(epoch) + ".pt"
        save_dict = {
            'epoch': epoch,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'info': info
            }
        torch.save(save_dict, path)
        print('saved training to ', path)

    def load_training(self, dirname, epoch, load_optimizer=True):
        path = dirname + "/snapshot" + str(epoch) + ".pt"
        print('load training from ', path)
        snapshot = torch.load(path)
        self.actor.load_state_dict(snapshot['actor_state_dict'])
        self.critic.load_state_dict(snapshot['critic_state_dict'])
        if load_optimizer:
            self.optimizer.load_state_dict(snapshot['optimizer_state_dict'])
        print('loaded epoch is {}.'.format(snapshot['epoch']))
        return snapshot['epoch'], snapshot['info']
