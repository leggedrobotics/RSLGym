import torch.nn as nn
import numpy as np
import torch
from torch.distributions import MultivariateNormal


class Actor:
    def __init__(self, architecture, distribution, obs_size, action_size, device='cpu'):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device
        self.obs_shape = [obs_size]
        self.action_shape = [action_size]

    def sample(self, obs):
        logits = self.architecture(obs)
        actions, log_prob = self.distribution.sample(logits)
        return actions.cpu().detach(), log_prob.cpu().detach()

    def evaluate(self, obs, actions):
        action_mean = self.architecture(obs)
        return self.distribution.evaluate(obs, action_mean, actions)

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def noiseless_action(self, obs):
        return self.architecture(obs)

    def save_deterministic_graph(self, example_input, file_name, device='cpu'):
        # example_input = torch.randn(1, self.architecture.input_shape[0]).to(device)
        transferred_graph = torch.jit.trace(self.architecture.to(device), example_input)
        torch.jit.save(transferred_graph, file_name)
        self.architecture.to(self.device)

    def state_dict(self):
        return {'architecture': self.architecture.state_dict(),
                'distribution': self.distribution.state_dict()}

    def load_state_dict(self, state_dict):
        self.architecture.load_state_dict(state_dict['architecture'])
        self.distribution.load_state_dict(state_dict['distribution'])


class Critic:
    def __init__(self, architecture, obs_size, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)
        self.obs_shape = [obs_size]

    def predict(self, obs):
        return self.architecture(obs).detach()

    def evaluate(self, obs):
        return self.architecture(obs)

    def parameters(self):
        return [*self.architecture.parameters()]

    def state_dict(self):
        return {'architecture': self.architecture.state_dict()}

    def load_state_dict(self, state_dict):
        self.architecture.load_state_dict(state_dict['architecture'])


class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.log_std = nn.Parameter(np.log(init_std) * torch.ones(dim))
        self.distribution = None

    def sample(self, logits):
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        self.distribution = MultivariateNormal(logits, covariance)

        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples)

        return samples, log_prob

    def evaluate(self, inputs, logits, outputs):
        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(logits, covariance)

        actions_log_prob = distribution.log_prob(outputs)
        entropy = distribution.entropy()

        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.log_std.detach().exp()
        new_log_std = torch.max(current_std, min_std.detach()).log().detach()
        self.log_std.data = new_log_std

    def enforce_maximum_std(self, max_std):
        current_std = self.log_std.detach().exp()
        new_log_std = torch.min(current_std, max_std.detach()).log().detach()
        self.log_std.data = new_log_std
