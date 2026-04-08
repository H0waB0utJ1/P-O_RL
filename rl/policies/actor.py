import torch
import torch.nn as nn
import torch.nn.functional as F
from rl.policies.base import Net, normc_fn

class Actor(Net):
    def __init__(self):
        super(Actor, self).__init__()

    def distribution(self, state, **kwargs):
        raise NotImplementedError

class Gaussian_FF_Actor(Actor):
    def __init__(
        self,
        state_dim,
        action_dim,
        layers=(256, 256),
        nonlinearity=F.relu,
        init_std=0.2,
        learn_std=True,
        bounded=False,
        normc_init=True,
    ):
        super(Gaussian_FF_Actor, self).__init__()

        self.actor_layers = nn.ModuleList()
        self.actor_layers.append(nn.Linear(state_dim, layers[0]))
        for i in range(len(layers) - 1):
            self.actor_layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.mean_head = nn.Linear(layers[-1], action_dim)

        if learn_std:
            self.log_std = nn.Parameter(torch.ones(action_dim) * torch.log(torch.tensor(init_std)))
        else:
            self.register_buffer("fixed_log_std",torch.ones(action_dim) * torch.log(torch.tensor(init_std)))
            self.log_std = None

        self.action_dim = action_dim
        self.nonlinearity = nonlinearity
        self.bounded = bounded
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        if normc_init:
            self.apply(normc_fn)
            with torch.no_grad():
                self.mean_head.weight.mul_(0.01)

    def dist_params(self, state):
        #state = self.normalize_state(state, update=self.training)

        x = state
        for l in self.actor_layers:
            x = self.nonlinearity(l(x))

        mu = self.mean_head(x)

        log_std = self.log_std if self.log_std is not None else self.fixed_log_std
        log_std = log_std.to(mu.device).expand_as(mu)
        std = torch.exp(log_std).clamp_min(1e-6)

        return mu, std, log_std

    def distribution(self, state):
        mu, std, log_std = self.dist_params(state)
        base_dist = torch.distributions.Normal(mu, std)

        if not self.bounded:
            return base_dist

        return torch.distributions.TransformedDistribution(
            base_dist,
            torch.distributions.transforms.TanhTransform(cache_size=1),
        )

    def forward(self, state, deterministic=True, return_log_prob=False):
        dist = self.distribution(state)

        if deterministic:
            action = dist.base_dist.mean if self.bounded else dist.mean
        else:
            action = dist.rsample()

        if return_log_prob:
            log_prob = dist.log_prob(action).sum(-1)
            return action, log_prob

        return action

class Gaussian_LSTM_Actor(Actor):
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        layers=(128, 128), 
        nonlinearity=torch.tanh, 
        init_std=0.2, 
        learn_std=False, 
        normc_init=False, 
        bounded=False
    ):
        super(Gaussian_LSTM_Actor, self).__init__()

        self.actor_layers = nn.ModuleList()
        self.actor_layers.append(nn.LSTMCell(state_dim, layers[0]))
        for i in range(len(layers) - 1):
            self.actor_layers.append(nn.LSTMCell(layers[i], layers[i + 1]))
        self.mean_head = nn.Linear(layers[-1], action_dim)

        self.action_dim = action_dim
        self.nonlinearity = nonlinearity
        self.bounded = bounded

        self.hidden = None
        self.cells = None

        self.state_dim = state_dim
        self.action_dim = action_dim

        if learn_std:
            self.log_std = nn.Parameter(torch.ones(action_dim) * torch.log(torch.tensor(init_std)))
        else:
            self.register_buffer("fixed_log_std",torch.ones(action_dim) * torch.log(torch.tensor(init_std)))
            self.log_std = None

        if normc_init:
            self.apply(normc_fn)
            with torch.no_grad():
                self.mean_head.weight.mul_(0.01)

    def get_hidden_state(self):
        return self.hidden, self.cells

    def init_hidden_state(self, batch_size=1, device=None, dtype=torch.float32):
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
        self.hidden = [torch.zeros(batch_size, l.hidden_size, device=device, dtype=dtype) for l in self.actor_layers]
        self.cells = [torch.zeros(batch_size, l.hidden_size, device=device, dtype=dtype) for l in self.actor_layers]

    def dist_params(self, state):
        #state = self.normalize_state(state, update=self.training)

        dims = len(state.size())
        x = state
        if dims == 3: # (t, b, )
            y = []
            for t, x_t in enumerate(x):
                for idx, layer in enumerate(self.actor_layers):
                    h, c = self.hidden[idx], self.cells[idx]
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]
                y.append(x_t)
            x = torch.stack(y)
        else: # 1 / 2
            if dims == 1: # (obs_dim)
                x = x.view(1, -1) # (1, obs_dim)
            for idx, layer in enumerate(self.actor_layers):
                h, c = self.hidden[idx], self.cells[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]
            if dims == 1:
                x = x.view(-1) # (obs_dim)

        mu = self.mean_head(x)

        log_std = self.log_std if self.log_std is not None else self.fixed_log_std
        log_std = log_std.to(mu.device).expand_as(mu)
        std = torch.exp(log_std).clamp_min(1e-6)

        return mu, std, log_std

    def distribution(self, state):
        mu, std, log_std = self.dist_params(state)
        base_dist = torch.distributions.Normal(mu, std)

        if not self.bounded:
            return base_dist

        return torch.distributions.TransformedDistribution(
            base_dist,
            torch.distributions.transforms.TanhTransform(cache_size=1),
        )

    def forward(self, state, deterministic=True, return_log_prob=False):
        dist = self.distribution(state)

        if deterministic:
            action = dist.base_dist.mean if self.bounded else dist.mean
        else:
            action = dist.rsample()

        if return_log_prob:
            log_prob = dist.log_prob(action).sum(-1)
            return action, log_prob

        return action