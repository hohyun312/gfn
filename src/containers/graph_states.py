from __future__ import annotations

import copy

import torch
import torch_geometric.data as gd
from torchtyping import TensorType

DonesTensor = TensorType["batch_shape", torch.bool]
RewardsTensor = TensorType["batch_shape", torch.float]


class GraphStates:
    s0 = gd.Data(x=torch.zeros(3))  # Source state of the DAG
    sf = gd.Data(x=torch.zeros(3))  # Dummy state, used to pad a batch of states

    def __init__(self, states_graph: gd.Batch):
        self.states_graph = states_graph
        self._log_rewards = (
            None  # Useful attribute if we want to store the log-reward of the states
        )

    @classmethod
    def make_states(cls, size: int, random: bool = False) -> GraphStates:
        """Create a GraphStates object with the given size, all initialized to s_0.
        If random is True, the states are initialized randomly. This requires that
        the environment implements the `make_random_states_tensor` class method.
        """
        if random:
            states_graph = cls.make_random_states_graph(size)
        else:
            states_graph = cls.make_initial_states_graph(size)

        return GraphStates(states_graph)

    @classmethod
    def make_initial_states_graph(cls, size: int) -> gd.Batch:
        states = gd.Batch.from_data_list([copy.deepcopy(cls.s0) for _ in range(size)])
        return states

    @classmethod
    def make_random_states_graph(cls, size: int) -> gd.Batch:
        raise NotImplementedError(
            "The environment does not support initialization of random states."
        )

    def __len__(self):
        return len(self.states_graph)

    def __repr__(self):
        return f"{self.__class__.__name__} object of batch size {len(self)}"

    @property
    def device(self) -> torch.device:
        return self.states_graph.x.device

    def __getitem__(self, index) -> GraphStates:
        """Access particular states of the batch."""
        # TODO: add more tests for this method
        states_list = self.states_graph[index]
        if not isinstance(states_list, list):
            states_list = [states_list]
        states_graph = gd.Batch.from_data_list(states_list)
        return self.__class__(states_graph)

    def extend(self, other: GraphStates) -> None:
        """Collates to another GraphStates object.

        Args:
            other (GraphStates): Batch of states to collate to.
        """
        states_list = (
            self.states_graph.to_data_list() + other.states_graph.to_data_list()
        )
        self.states_graph = gd.Batch.from_data_list(states_list)

    def extend_with_sf(self, size: int) -> None:
        """Extends to a GraphStates object by adding `s_f.
        This is useful to extend trajectories of different lengths."""
        assert (
            len(self) <= size
        ), "`size` should be equal or greater than the current batch size."
        sf_states = [copy.deepcopy(self.__class__.sf) for _ in range(size - len(self))]
        states_list = self.states_graph.to_data_list() + sf_states
        self.states_graph = gd.Batch.from_data_list(states_list)

    def compare(self, other: GraphStates):
        """Given a GraphStates object, returns a tensor of booleans indicating whether the states
        are equal to the states in self.

        Args:
            other (GraphStates): GraphStates object to compare to.

        Returns:
            DonesTensor: Tensor of booleans indicating whether the states are equal to the states in self.

        Warning:
            It is not permutation invariant
        """
        # TODO: improve the compare logic.
        assert len(self) == len(other)
        data_zip = zip(
            self.states_graph.to_data_list(), other.states_graph.to_data_list()
        )
        values = []
        for self_g, self_g in data_zip:
            attrs = list(self_g._store)
            value = True
            for attr in attrs:
                value = value and all(self_g._store[attr] == self_g._store[attr])

            values.append(value)
        out = torch.BoolTensor(values)
        return out

    @property
    def is_initial_state(self) -> DonesTensor:
        r"""Return a boolean tensor of shape=(*batch_shape,),
        where True means that the state is $s_0$ of the DAG.
        """
        s0_states = [copy.deepcopy(self.__class__.s0) for _ in range(len(self))]
        source_states = GraphStates(gd.Batch.from_data_list(s0_states))
        return self.compare(source_states)

    @property
    def is_sink_state(self) -> DonesTensor:
        r"""Return a boolean tensor of shape=(*batch_shape,),
        where True means that the state is $s_f$ of the DAG.
        """
        sf_states = [copy.deepcopy(self.__class__.sf) for _ in range(len(self))]
        sink_states = GraphStates(gd.Batch.from_data_list(sf_states))
        return self.compare(sink_states)

    @property
    def log_rewards(self) -> RewardsTensor:
        return self._log_rewards

    @log_rewards.setter
    def log_rewards(self, log_rewards: RewardsTensor) -> None:
        self._log_rewards = log_rewards


if __name__ == "__main__":
    states = GraphStates.make_states(4)
    states.__repr__()
    states.__len__()
    states.device

    states.extend_with_sf(10)
    assert len(states) == 10
    assert True == all(states.compare(states))
