from __future__ import annotations

import copy
from math import prod
from typing import ClassVar
import numpy as np


class NumpyStates:
    state_shape: ClassVar[tuple[int, ...]] = (1,)  # Shape of one state
    s0: ClassVar[np.ndarray] = np.zeros(4)  # Source state of the DAG
    sf: ClassVar[np.ndarray] = np.ones(4)  # Dummy state, used to pad a batch of states

    def __init__(self, states_array: np.ndarray):
        self.states_array = states_array
        self.batch_shape = tuple(self.states_array.shape)[: -len(self.state_shape)]
        self._log_rewards = (
            None  # Useful attribute if we want to store the log-reward of the states
        )

    @classmethod
    def from_batch_shape(
        cls, batch_shape: tuple[int, ...], random: bool = False
    ) -> NumpyStates:
        """Create a GraphStates object with the given size, all initialized to s_0.
        If random is True, the states are initialized randomly. This requires that
        the environment implements the `make_random_states_tensor` class method.
        """
        if random:
            states_array = cls.make_random_states_array(batch_shape)
        else:
            states_array = cls.make_initial_states_array(batch_shape)

        return NumpyStates(states_array)

    @classmethod
    def make_initial_states_array(cls, batch_shape: tuple[int, ...]) -> np.ndarray:
        state_ndim = len(cls.state_shape)
        assert cls.s0 is not None and state_ndim is not None
        return np.tile(cls.s0, batch_shape + ((1,) * state_ndim))

    @classmethod
    def make_random_states_array(cls, batch_shape: tuple[int, ...]) -> np.ndarray:
        raise NotImplementedError(
            "The environment does not support initialization of random states."
        )

    def __len__(self):
        return prod(self.batch_shape)

    def __repr__(self):
        return f"{self.__class__.__name__} object of batch shape {self.batch_shape} and state shape {self.state_shape}"

    def __getitem__(self, index) -> NumpyStates:
        """Access particular states of the batch."""
        states = self.states_array[index]
        return self.__class__(states)

    def extend(self, other: NumpyStates) -> None:
        """Collates to another NumpyStates object of the same batch shape, which should be 1 or 2.

        Args:
            other (States): Batch of states to collate to.

        Raises:
            ValueError: if self.batch_shape != other.batch_shape or if self.batch_shape != (1,) or (2,)
        """
        other_batch_shape = other.batch_shape
        if len(other_batch_shape) == len(self.batch_shape) == 1:
            # This corresponds to adding a state to a trajectory
            self.batch_shape = (self.batch_shape[0] + other_batch_shape[0],)
            self.states_array = np.concatenate(
                (self.states_array, other.states_array), axis=0
            )
        elif len(other_batch_shape) == len(self.batch_shape) == 2:
            # This corresponds to adding a trajectory to a batch of trajectories
            self.extend_with_sf(
                required_first_dim=max(self.batch_shape[0], other_batch_shape[0])
            )
            other.extend_with_sf(
                required_first_dim=max(self.batch_shape[0], other_batch_shape[0])
            )
            self.batch_shape = (
                self.batch_shape[0],
                self.batch_shape[1] + other_batch_shape[1],
            )
            self.states_array = np.concatenate(
                (self.states_array, other.states_array), axis=1
            )
        else:
            raise ValueError(
                f"extend is not implemented for batch shapes {self.batch_shape} and {other_batch_shape}"
            )

    def extend_with_sf(self, required_first_dim: int) -> None:
        """Takes a two-dimensional batch of states (i.e. of batch_shape (a, b)),
        and extends it to a NumpyStates object of batch_shape (required_first_dim, b),
        by adding the required number of `s_f` tensors. This is useful to extend trajectories
        of different lengths."""
        assert (
            len(self.batch_shape) == 2
        ), f"extend_with_sf is only implemented for two-dimensional batch shape"
        if self.batch_shape[0] >= required_first_dim:
            return

        tile_shape = (
            required_first_dim - self.batch_shape[0],
            self.batch_shape[1],
            1,
        )
        sf_states = np.tile(self.__class__.sf, tile_shape)
        self.states_array = np.concatenate([self.states_array, sf_states], axis=0)
        self.batch_shape = (required_first_dim, self.batch_shape[1])

    def compare(self, other: np.ndarray) -> np.ndarray:
        """Given a tensor of states, returns a tensor of booleans indicating whether the states
        are equal to the states in self.

        Args:
            other (StatesTensor): Tensor of states to compare to.

        Returns:
            DonesTensor: Tensor of booleans indicating whether the states are equal to the states in self.
        """
        out = self.states_array == other
        state_ndim = len(self.__class__.state_shape)
        for _ in range(state_ndim):
            out = out.all(axis=-1)
        return out

    @property
    def is_initial_state(self) -> np.ndarray:
        r"""Return a boolean tensor of shape=(*batch_shape,),
        where True means that the state is $s_0$ of the DAG.
        """
        state_ndim = len(self.__class__.state_shape)
        source_states_array = np.tile(
            self.__class__.s0, (self.batch_shape + (1,) * state_ndim)
        )
        return self.compare(source_states_array)

    @property
    def is_sink_state(self) -> np.ndarray:
        r"""Return a boolean tensor of shape=(*batch_shape,),
        where True means that the state is $s_f$ of the DAG.
        """
        state_ndim = len(self.__class__.state_shape)
        sink_states_array = np.tile(
            self.__class__.sf, (self.batch_shape + (1,) * state_ndim)
        )
        return self.compare(sink_states_array)

    @property
    def log_rewards(self) -> np.ndarray | None:
        return self._log_rewards

    @log_rewards.setter
    def log_rewards(self, log_rewards: np.ndarray) -> None:
        self._log_rewards = log_rewards


if __name__ == "__main__":
    states = NumpyStates.from_batch_shape((3, 4))
    states.__repr__()
    assert states.__len__() == 12

    states.extend_with_sf(required_first_dim=10)
    assert states.batch_shape == (10, 4)

    states.extend_with_sf(required_first_dim=12)
    assert states.batch_shape == (12, 4)

    states.extend_with_sf(required_first_dim=3)
    assert states.batch_shape == (12, 4)

    other_states = NumpyStates.from_batch_shape((12, 7))
    states.extend(other_states)
    assert states.batch_shape == (12, 11)

    states = NumpyStates.from_batch_shape((3, 4))
    assert states.is_initial_state.all() == True
