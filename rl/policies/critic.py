import torch
import torch.nn as nn
from rl.policies.base import Net, normc_fn

class Critic(Net):
    def __init__(self):
        super(Critic, self).__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class FF_V(Critic):
    def __init__(self, state_dim, layers=(256, 256), nonlinearity=nn.ReLU(), normc_init=True):
        super(FF_V, self).__init__()

        self.critic_layers = nn.ModuleList()
        self.critic_layers.append(nn.Linear(state_dim, layers[0]))
        for i in range(len(layers) - 1):
            self.critic_layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.network_out = nn.Linear(layers[-1], 1)

        self.nonlinearity = nonlinearity

        if normc_init:
            self.apply(normc_fn)

    def forward(self, state):
        #state = self.normalize_state(state, update=self.training)

        x = state
        for l in self.critic_layers:
            x = self.nonlinearity(l(x))
        value = self.network_out(x)
        return value

class LSTM_V(Critic):
    def __init__(self, state_dim, layers=(128, 128), normc_init=True):
        super(LSTM_V, self).__init__()

        self.critic_layers = nn.ModuleList()
        self.critic_layers.append(nn.LSTMCell(state_dim, layers[0]))
        for i in range(len(layers) - 1):
            self.critic_layers.append(nn.LSTMCell(layers[i], layers[i + 1]))
        self.network_out = nn.Linear(layers[-1], 1)

        if normc_init:
            self.apply(normc_fn)

    def get_hidden_state(self):
        return self.hidden, self.cells

    def init_hidden_state(self, batch_size=1, device=None, dtype=torch.float32):
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
        self.hidden = [torch.zeros(batch_size, l.hidden_size, device=device, dtype=dtype) for l in self.critic_layers]
        self.cells = [torch.zeros(batch_size, l.hidden_size, device=device, dtype=dtype) for l in self.critic_layers]

    def forward(self, state):
        #state = self.normalize_state(state, update=self.training)

        dims = len(state.size())
        if dims == 3:
            values = []
            for t, x_t in enumerate(state):
                for idx, layer in enumerate(self.critic_layers):
                    c, h = self.cells[idx], self.hidden[idx]
                    self.hidden[idx], self.cells[idx] = layer(x_t, (h, c))
                    x_t = self.hidden[idx]
                val = self.network_out(x_t)
                values.append(val)
            return torch.stack(values)
        else:
            if dims == 1:
                x = state.view(1, -1)
            else:
                x = state
            for idx, layer in enumerate(self.critic_layers):
                c, h = self.cells[idx], self.hidden[idx]
                self.hidden[idx], self.cells[idx] = layer(x, (h, c))
                x = self.hidden[idx]
            x = self.network_out(x)
            if dims == 1:
                return x.squeeze(0)
            
            return x