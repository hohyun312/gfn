from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal
import enum
from rdkit import Chem

import torch
from src.containers.numpy_states import NumpyStates

from torchtyping import TensorType


# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]


class StatesType(enum.Enum):
    INTERNAL = enum.auto()
    SOURCE = enum.auto()
    SINK = enum.auto()


@dataclass
class MolData:
    x: Optional[torch.Tensor] = None
    edge_index: Optional[torch.Tensor] = None
    edge_attr: Optional[torch.Tensor] = None
    stype: Optional[StatesType] = None
    device: torch.device = torch.device("cpu")
    smiles: "str" = None
    mol: Chem.Mol = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MolData):
            return NotImplemented
        return (self.stype == other.stype) and (self.smiles == other.smiles)

    def __repr__(self):
        return str(self.smiles)


class GraphStates(NumpyStates):
    state_shape = (1,)
    s0 = MolData(stype=StatesType.SOURCE, smiles="C", mol=Chem.MolFromSmiles("C"))
    sf = MolData(stype=StatesType.SINK, smiles="", mol=Chem.MolFromSmiles(""))


class GraphEnv:
    def __init__(self, device_str: Literal["cpu", "cuda"] = "cpu"):
        self.s0 = GraphStates.s0
        self.sf = GraphStates.sf
        self.device = torch.device(device_str)
        # TODO: self.action_space

    def make_States_class(self) -> GraphStates:
        return GraphStates

    def log_reward(self, final_states: GraphStates) -> TensorFloat:
        raise NotImplementedError

    def is_exit_actions(self, actions: TensorLong) -> TensorBool:
        raise NotImplementedError

    def maskless_step(self, states: GraphStates, actions: TensorLong) -> None:
        raise NotImplementedError

    def maskless_backward_step(self, states: GraphStates, actions: TensorLong) -> None:
        raise NotImplementedError
