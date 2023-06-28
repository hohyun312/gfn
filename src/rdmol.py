from typing import Literal, Callable, ClassVar
from torchtyping import TensorType

from gfn.containers.states import States
from gfn.envs.env import Env

import torch
import torch_geometric.data as gd
from rdkit import Chem

import numpy as np


# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]


class RdMolEnv(Env):
    def __init__(
        self,
        mol2graph: Callable[[Chem.Mol], gd.Data],
        device_str: Literal["cpu", "cuda"] = "cpu",
    ):
        self.mol2graph = mol2graph
        self.s0 = np.array(Chem.MolFromSmiles("C"), dtype=object)
        self.sf = np.array(-1, dtype=object)

        self.action_space = None
        self.device = torch.device(device_str)
        self.States = self.make_States_class()

    def make_States_class(self):
        "Creates a States class for this environment"
        setattr(NumpyStates, "state_shape", (1,))
        setattr(NumpyStates, "s0", self.s0)
        setattr(NumpyStates, "sf", self.sf)
        return NumpyStates

    def log_reward(self, final_states: States) -> TensorFloat:
        device = final_states.states_tensor.device
        rew_states = 1e-8 + torch.arange(self.length, device=device)
        return final_states.states_tensor.float() @ rew_states

    def is_exit_actions(self, actions: TensorLong) -> TensorBool:
        return actions == self.action_space.n - 1

    def maskless_backward_step(self):
        pass

    def maskless_step(self):
        pass


class NumpyStates(States):
    state_shape: ClassVar[tuple[int, ...]]  # Shape of one state
    s0: ClassVar[np.ndarray]  # Source state of the DAG
    sf: ClassVar[np.ndarray]  # Dummy state, used to pad a batch of states

    def __init__(self, states_tensor: np.ndarray):
        self.states_tensor = states_tensor
        self.batch_shape = tuple(self.states_tensor.shape)[: -len(self.state_shape)]
        self.forward_masks, self.backward_masks = None, None

        self._log_rewards = (
            None  # Useful attribute if we want to store the log-reward of the states
        )

    @classmethod
    def make_initial_states_tensor(cls, batch_shape: tuple[int]) -> np.ndarray:
        state_ndim = len(cls.state_shape)
        assert cls.s0 is not None and state_ndim is not None
        return np.tile(cls.s0, batch_shape + ((1,) * state_ndim))

    @property
    def device(self) -> torch.device:
        return self.states_tensor.device


if __name__ == "__main__":
    env = RdMolEnv(None)
    states = env.make_States_class()
    initial_states = states.from_batch_shape((3, 4))
    print(
        initial_states
    )  # NumpyStates object of batch shape (3, 4) and state shape (1,)
