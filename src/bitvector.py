from typing import ClassVar, Literal, Tuple, cast
import itertools

import torch
from gymnasium.spaces import Discrete
from torchtyping import TensorType

from gfn.containers.states import States
from gfn.envs.env import Env

# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]
ForwardMasksTensor = TensorType["batch_shape", "n_actions", torch.bool]
BackwardMasksTensor = TensorType["batch_shape", "n_actions - 1", torch.bool]
OneStateTensor = TensorType["state_shape", torch.float]
StatesTensor = TensorType["batch_shape", "state_shape", torch.float]


class BitVector(Env):
    def __init__(self, length: int = 4, device_str: Literal["cpu", "cuda"] = "cpu"):
        self.length = length

        s0 = torch.zeros(length, dtype=torch.long, device=torch.device(device_str))
        sf = torch.full(
            (length,), fill_value=-1, dtype=torch.long, device=torch.device(device_str)
        )

        action_space = Discrete(length + 1)

        super().__init__(
            action_space=action_space,
            s0=s0,
            sf=sf,
            device_str=device_str,
        )

    def make_States_class(self) -> type[States]:
        "Creates a States class for this environment"
        env = self

        class BitVectorStates(States):
            state_shape: ClassVar[tuple[int, ...]] = (env.length,)
            s0 = env.s0
            sf = env.sf

            @classmethod
            def make_random_states_tensor(
                cls, batch_shape: Tuple[int, ...]
            ) -> StatesTensor:
                "Creates a batch of random states."
                states_tensor = torch.randint(
                    0, 2, batch_shape + env.s0.shape, device=env.device
                )
                return states_tensor

            def make_masks(self) -> Tuple[ForwardMasksTensor, BackwardMasksTensor]:
                "Mask illegal (forward and backward) actions."
                forward_masks = torch.ones(
                    (*self.batch_shape, env.n_actions),
                    dtype=torch.bool,
                    device=env.device,
                )
                backward_masks = torch.ones(
                    (*self.batch_shape, env.n_actions - 1),
                    dtype=torch.bool,
                    device=env.device,
                )

                return forward_masks, backward_masks

            def update_masks(self) -> None:
                "Update the masks based on the current states."
                # The following two lines are for typing only.
                self.forward_masks = cast(ForwardMasksTensor, self.forward_masks)
                self.backward_masks = cast(BackwardMasksTensor, self.backward_masks)

                self.forward_masks[..., :-1] = self.states_tensor == 0
                self.backward_masks = self.states_tensor == 1

        return BitVectorStates

    def log_reward(self, final_states: States) -> TensorFloat:
        device = final_states.states_tensor.device
        rew_states = 1e-8 + torch.arange(self.length, device=device)
        return final_states.states_tensor.float() @ rew_states

    def is_exit_actions(self, actions: TensorLong) -> TensorBool:
        return actions == self.action_space.n - 1

    def maskless_step(self, states: StatesTensor, actions: TensorLong) -> None:
        states.scatter_(-1, actions.unsqueeze(-1), 1, reduce="add")

    def maskless_backward_step(self, states: StatesTensor, actions: TensorLong) -> None:
        states.scatter_(-1, actions.unsqueeze(-1), -1, reduce="add")

    @property
    def all_states(self) -> States:
        all_states_list = list(itertools.product((0, 1), repeat=4))
        all_states_tensor = torch.LongTensor(all_states_list)
        return self.States(all_states_tensor)
