import torch
import torch.nn as nn

def normc_fn(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        with torch.no_grad():
            m.weight.normal_(0, 1)
            m.weight.mul_(1.0 / torch.sqrt(m.weight.pow(2).sum(dim=1, keepdim=True)))
            if m.bias is not None:
                m.bias.zero_()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Welford running statistics (state normalization)
        self.register_buffer("welford_state_mean", torch.zeros(1))
        self.register_buffer("welford_state_M2", torch.zeros(1))
        self.register_buffer("welford_state_n", torch.tensor(0.0))

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def normalize_state(self, state, update=True):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        # flatten all leading dims, keep last dim as feature
        orig_shape = state.shape
        state = state.view(-1, orig_shape[-1])

        device = state.device
        dtype = state.dtype

        # lazy init statistics dimension
        if self.welford_state_n.item() == 0:
            self.welford_state_mean = torch.zeros(
                state.size(1), device=device, dtype=dtype
            )
            self.welford_state_M2 = torch.zeros(
                state.size(1), device=device, dtype=dtype
            )
            self.welford_state_n = torch.tensor(
                0.0, device=device, dtype=dtype
            )

        if update:
            with torch.no_grad():
                batch_n = state.size(0)
                batch_mean = state.mean(dim=0)
                batch_var = state.var(dim=0, unbiased=False)

                delta = batch_mean - self.welford_state_mean
                total_n = self.welford_state_n + batch_n

                new_mean = self.welford_state_mean + delta * batch_n / total_n
                new_M2 = (
                    self.welford_state_M2
                    + batch_var * batch_n
                    + delta.pow(2) * self.welford_state_n * batch_n / total_n
                )

                self.welford_state_mean.copy_(new_mean)
                self.welford_state_M2.copy_(new_M2)
                self.welford_state_n.copy_(total_n)

        var = self.welford_state_M2 / self.welford_state_n.clamp_min(1.0)
        std = torch.sqrt(var + 1e-8)

        state = (state - self.welford_state_mean) / std
        return state.view(orig_shape)

    def copy_normalizer_stats(self, net):
        with torch.no_grad():
            self.welford_state_mean.copy_(net.welford_state_mean)
            self.welford_state_M2.copy_(net.welford_state_M2)
            self.welford_state_n.copy_(net.welford_state_n)

    def initialize_parameters(self):
        self.apply(normc_fn)