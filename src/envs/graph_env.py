from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal
import enum
from rdkit import Chem

import torch
from src.containers.numpy_states import NumpyStates

from torchtyping import TensorType

import numpy as np
from collections import deque
from functools import cached_property
import datamol as dm


# Typing
TensorLong = TensorType["batch_shape", torch.long]
TensorFloat = TensorType["batch_shape", torch.float]
TensorBool = TensorType["batch_shape", torch.bool]


class SubActionType(enum.Enum):
    Meta = enum.auto()
    InNode = enum.auto()
    OutNode = enum.auto()
    Edge = enum.auto()


@dataclass
class SubAction:
    atype: Optional[SubActionType] = None
    index: Optional[int] = None
    logprob: Optional[float] = None
    n_actions: Optional[int] = None


class MetaActionType(enum.Enum):
    Stop = enum.auto()
    AddNode = enum.auto()
    AddEdge = enum.auto()


@dataclass
class CompositeAction:
    atype: Optional[MetaActionType] = None
    meta_action: Optional[SubAction] = None
    node1_action: Optional[SubAction] = None
    node2_action: Optional[SubAction] = None
    edge_action: Optional[SubAction] = None


class StateType(enum.Enum):
    Source = enum.auto()
    Internal = enum.auto()
    Sink = enum.auto()


@dataclass
class MolState:
    id2atom = ["C", "N", "O", "S", "P", "F", "I", "Cl", "Br"]
    id2bond = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]

    stype: Optional[StateType] = None
    x: Optional[torch.Tensor] = None
    edge_index: Optional[torch.Tensor] = None
    edge_attr: Optional[torch.Tensor] = None
    smiles: Optional[str] = None
    mol: Optional[Chem.Mol] = None
    device: torch.device = torch.device("cpu")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MolState):
            return NotImplemented
        return (self.stype == other.stype) and self.hash == other.hash

    def __repr__(self):
        return f'MolState("{self.smiles}")'

    @cached_property
    def hash(self):
        return dm.hash_mol(self.mol)

    @classmethod
    def from_smiles(cls, smiles):
        mol = Chem.MolFromSmiles(smiles)
        return cls(mol=mol, smiles=smiles)

    @classmethod
    def from_molecule(cls, mol):
        smiles = Chem.MolToSmiles(mol)
        return cls(mol=mol, smiles=smiles)

    def apply_action(self, action):
        mol = Chem.RWMol(self.mol)
        if action.atype == MetaActionType.AddNode:
            node2_idx = mol.AddAtom(Chem.Atom(self.id2atom[action.node2_action.index]))
            mol.AddBond(
                action.node1_action.index,
                node2_idx,
                order=self.id2bond[action.edge_action.index],
            )
        elif action.atype == MetaActionType.AddEdge:
            mol.AddBond(
                action.node1_action.index,
                action.node2_action.index,
                order=self.id2bond[action.edge_action.index],
            )
        elif action.atype == MetaActionType.Stop:
            pass
        else:
            raise ValueError(action.atype)
        return self.__class__.from_molecule(mol)


def mol_to_trajectory(mol, atom2id, bond2id):
    # choose random atom
    rand_idx = np.random.randint(mol.GetNumAtoms())
    init_atom = mol.GetAtomWithIdx(rand_idx)

    # variables for breath-first search
    queue = deque([init_atom.GetIdx()])
    visited_atoms = set([init_atom.GetIdx()])
    visited_bonds = set()
    atom_mapping = {init_atom.GetIdx(): 0}

    # MolState object to be returned
    rwmol = Chem.RWMol()
    rwmol.AddAtom(Chem.Atom(init_atom.GetSymbol()))
    state = MolState.from_molecule(rwmol)
    trajectory = [state]
    while queue:
        i = queue.popleft()
        atom_i = mol.GetAtomWithIdx(i)
        for atom_j in atom_i.GetNeighbors():
            j = atom_j.GetIdx()
            if j in visited_atoms:
                action_type = MetaActionType.AddEdge
                node2_action = SubAction(
                    atype=SubActionType.InNode, index=atom_mapping[j]
                )
            else:
                atom_mapping[j] = len(visited_atoms)
                queue.append(j)
                visited_atoms.add(j)
                action_type = MetaActionType.AddNode
                node2_action = SubAction(
                    atype=SubActionType.OutNode, index=atom2id[atom_j.GetSymbol()]
                )

            edge_tuple = (i, j) if i < j else (j, i)
            if edge_tuple not in visited_bonds:
                visited_bonds.add(edge_tuple)
                bond = mol.GetBondBetweenAtoms(i, j)
                bondtype = str(bond.GetBondType())
                action = CompositeAction(
                    atype=action_type,
                    meta_action=SubAction(
                        atype=SubActionType.Meta, index=action_type.value
                    ),
                    node1_action=SubAction(
                        atype=SubActionType.InNode, index=atom_mapping[i]
                    ),
                    node2_action=node2_action,
                    edge_action=SubAction(
                        atype=SubActionType.Edge, index=bond2id[bondtype]
                    ),
                )
                trajectory += [action]
                state = state.apply_action(action)
                trajectory += [state]
    action = CompositeAction(
        atype=MetaActionType.Stop,
        meta_action=SubAction(
            atype=SubActionType.Meta, index=MetaActionType.Stop.value
        ),
    )
    trajectory += [action]
    state = state.apply_action(action)
    trajectory += [state]
    return trajectory


def trajectory_to_mol(init_state, actions, id2atom, id2bond):
    mol = Chem.RWMol(init_state.mol)
    for action in actions:
        if action.atype == MetaActionType.AddNode:
            node2_idx = mol.AddAtom(Chem.Atom(id2atom[action.node2_action.index]))
            mol.AddBond(
                action.node1_action.index,
                node2_idx,
                order=id2bond[action.edge_action.index],
            )
        elif action.atype == MetaActionType.AddEdge:
            mol.AddBond(
                action.node1_action.index,
                action.node2_action.index,
                order=id2bond[action.edge_action.index],
            )
        elif action.atype == MetaActionType.Stop:
            break
        else:
            raise ValueError(action.atype)
    return mol


class BatchedMolStates(NumpyStates):
    state_shape = (1,)
    s0 = MolState(stype=StateType.Source, smiles="C", mol=Chem.MolFromSmiles("C"))
    sf = MolState(stype=StateType.Sink, smiles="", mol=Chem.MolFromSmiles(""))


class MolEnv:
    def __init__(self, device_str: Literal["cpu", "cuda"] = "cpu"):
        self.s0 = BatchedMolStates.s0
        self.sf = BatchedMolStates.sf
        self.device = torch.device(device_str)
        # TODO: self.action_space

    def make_States_class(self) -> BatchedMolStates:
        return BatchedMolStates

    def log_reward(self, final_states: BatchedMolStates) -> TensorFloat:
        raise NotImplementedError

    def is_exit_actions(self, actions: TensorLong) -> TensorBool:
        raise NotImplementedError

    def maskless_step(self, states: BatchedMolStates, actions: CompositeAction) -> None:
        raise NotImplementedError

    def maskless_backward_step(
        self, states: BatchedMolStates, actions: CompositeAction
    ) -> None:
        raise NotImplementedError


if __name__ == "__main__":
    atom2id = {"C": 0, "N": 1, "O": 2, "S": 3, "P": 4, "F": 5, "I": 6, "Cl": 7, "Br": 8}
    bond2id = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}

    id2atom = ["C", "N", "O", "S", "P", "F", "I", "Cl", "Br"]
    id2bond = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]

    mol = Chem.MolFromSmiles("CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1")
    traj = mol_to_trajectory(mol, atom2id, bond2id)
    mol = trajectory_to_mol(traj[0], traj[1::2], id2atom, id2bond)
