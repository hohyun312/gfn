import torch
from torch import nn
from torch.distributions import Categorical


class TBModel(nn.Module):
    def __init__(self, ndim, hidden_dims=[20, 20], uniform_backward_flow=False):
        super().__init__()
        self.ndim = ndim
        self.uniform_backward_flow = uniform_backward_flow
        if uniform_backward_flow:
            output_dim = self.ndim
        else:
            output_dim = 2 * self.ndim
        dims = [self.ndim] + hidden_dims + [output_dim]
        modules = []
        for i in range(len(dims) - 1):
            modules += [nn.Linear(dims[i], dims[i + 1])]
            modules += [nn.LeakyReLU()]
        modules.pop()
        self.mlp = nn.Sequential(*modules)
        self.logZ = nn.Parameter(torch.ones(1))

    def forward(self, states):
        x = torch.tensor(states).float().to(self.device)
        logits = self.mlp(x)
        if not self.uniform_backward_flow:
            P_F, P_B = logits.chunk(2, dim=-1)
        else:
            P_F = logits
            P_B = torch.ones_like(logits)
            
        P_F = P_F * (1 - x) + x * -100
        P_B = P_B * x + (1 - x) * -100
        return P_F, P_B

    @torch.no_grad()
    def act(self, states):
        P_F, P_B = self(states)
        return Categorical(logits=P_F).sample()

    @property
    def device(self):
        return next(self.parameters()).device