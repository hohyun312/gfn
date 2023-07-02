from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from gfn.envs import Env
    from src.containers.numpy_states import NumpyStates

import numpy as np

# from gfn.containers.transitions import Transitions


class Trajectories:
    def __init__(
        self,
        env: Env,
        states: NumpyStates | None = None,
        actions: np.ndarray | None = None,
        when_is_done: np.ndarray | None = None,
        is_backward: bool = False,
        log_rewards: np.ndarray | None = None,
        log_probs: np.ndarray | None = None,
    ) -> None:
        """Container for complete trajectories (starting in s_0 and ending in s_f).
        Trajectories are represented as a States object with bi-dimensional batch shape.
        The first dimension represents the time step, the second dimension represents the trajectory index.
        Because different trajectories may have different lengths, shorter trajectories are padded with
        the tensor representation of the terminal state (s_f or s_0 depending on the direction of the trajectory), and
        actions is appended with -1's.
        The actions are represented as a two dimensional tensor with the first dimension representing the time step
        and the second dimension representing the trajectory index.
        The when_is_done tensor represents the time step at which each trajectory ends.


        Args:
            env (Env): The environment in which the trajectories are defined.
            states (States, optional): The states of the trajectories. Defaults to None.
            actions (Tensor2D, optional): The actions of the trajectories. Defaults to None.
            when_is_done (Tensor1D, optional): The time step at which each trajectory ends. Defaults to None.
            is_backward (bool, optional): Whether the trajectories are backward or forward. Defaults to False.
            log_rewards (FloatTensor1D, optional): The log_rewards of the trajectories. Defaults to None.
            log_probs (FloatTensor2D, optional): The log probabilities of the trajectories' actions. Defaults to None.

        If states is None, then the states are initialized to an empty States object, that can be populated on the fly.
        If log_rewards is None, then `env.log_reward` is used to compute the rewards, at each call of self.log_rewards
        """
        self.env = env
        self.is_backward = is_backward
        self.states = (
            states
            if states is not None
            else env.States.from_batch_shape(batch_shape=(0, 0))
        )
        assert len(self.states.batch_shape) == 2
        self.actions = (
            actions
            if actions is not None
            else np.full(shape=(0, 0), fill_value=-1, dtype=np.int64)
        )
        self.when_is_done = (
            when_is_done
            if when_is_done is not None
            else np.full(shape=(0,), fill_value=-1, dtype=np.int64)
        )
        self._log_rewards = log_rewards
        self.log_probs = (
            log_probs
            if log_probs is not None
            else np.full(shape=(0, 0), fill_value=0, dtype=np.float32)
        )

    def __repr__(self) -> str:
        return f"Trajectories(n_trajectories={self.n_trajectories}, max_length={self.max_length}"

    @property
    def n_trajectories(self) -> int:
        return self.states.batch_shape[1]

    def __len__(self) -> int:
        return self.n_trajectories

    @property
    def max_length(self) -> int:
        if len(self) == 0:
            return 0

        return self.actions.shape[0]

    @property
    def last_states(self) -> States:
        return self.states[self.when_is_done - 1, np.arange(self.n_trajectories)]

    @property
    def log_rewards(self) -> np.ndarray | None:
        if self._log_rewards is not None:
            assert self._log_rewards.shape == (self.n_trajectories,)
            return self._log_rewards
        if self.is_backward:
            return None
        try:
            return self.env.log_reward(self.last_states)
        except NotImplementedError:
            return np.log(self.env.reward(self.last_states))

    def __getitem__(self, index: int | Sequence[int]) -> Trajectories:
        "Returns a subset of the `n_trajectories` trajectories."
        if isinstance(index, int):
            index = [index]
        when_is_done = self.when_is_done[index]
        new_max_length = when_is_done.max().item() if len(when_is_done) > 0 else 0
        states = self.states[:, index]
        actions = self.actions[:, index]
        log_probs = self.log_probs[:, index]
        states = states[: 1 + new_max_length]
        actions = actions[:new_max_length]
        log_probs = log_probs[:new_max_length]
        log_rewards = (
            self._log_rewards[index] if self._log_rewards is not None else None
        )

        return Trajectories(
            env=self.env,
            states=states,
            actions=actions,
            when_is_done=when_is_done,
            is_backward=self.is_backward,
            log_rewards=log_rewards,
            log_probs=log_probs,
        )

    def extend(self, other: Trajectories) -> None:
        """Extend the trajectories with another set of trajectories."""
        self.extend_actions(required_first_dim=max(self.max_length, other.max_length))
        other.extend_actions(required_first_dim=max(self.max_length, other.max_length))

        self.states.extend(other.states)
        self.actions = np.concatenate((self.actions, other.actions), axis=1)
        self.when_is_done = np.concatenate(
            (self.when_is_done, other.when_is_done), axis=0
        )
        self.log_probs = np.concatenate((self.log_probs, other.log_probs), axis=1)

        if self._log_rewards is not None and other._log_rewards is not None:
            self._log_rewards = np.concatenate(
                (self._log_rewards, other._log_rewards), axis=0
            )
        else:
            self._log_rewards = None

    def extend_actions(self, required_first_dim: int) -> None:
        """Extends the actions and log_probs along the first dimension by by adding -1s as necessary.
        This is useful for extending trajectories of different lengths."""
        if self.max_length >= required_first_dim:
            return
        self.actions = np.concatenate(
            (
                self.actions,
                np.full(
                    shape=(
                        required_first_dim - self.actions.shape[0],
                        self.n_trajectories,
                    ),
                    fill_value=-1,
                    dtype=np.int64,
                ),
            ),
            axis=0,
        )
        self.log_probs = np.concatenate(
            (
                self.log_probs,
                np.full(
                    shape=(
                        required_first_dim - self.log_probs.shape[0],
                        self.n_trajectories,
                    ),
                    fill_value=0,
                    dtype=np.float32,
                ),
            ),
            axis=0,
        )

    @staticmethod
    def revert_backward_trajectories(trajectories: Trajectories) -> Trajectories:
        assert trajectories.is_backward
        new_actions = np.full_like(trajectories.actions, -1)
        new_actions = np.concatenate(
            [new_actions, np.full((1, len(trajectories)), -1)], axis=0
        )
        new_states = np.tile(
            trajectories.env.sf,
            (trajectories.when_is_done.max() + 1, len(trajectories), 1),
        )
        new_when_is_done = trajectories.when_is_done + 1
        for i in range(len(trajectories)):
            new_actions[trajectories.when_is_done[i], i] = (
                trajectories.env.n_actions - 1
            )
            new_actions[: trajectories.when_is_done[i], i] = np.flip(
                trajectories.actions[: trajectories.when_is_done[i], i], axis=0
            )
            new_states[: trajectories.when_is_done[i] + 1, i] = np.flip(
                trajectories.states.states_tensor[
                    : trajectories.when_is_done[i] + 1, i
                ],
                axis=0,
            )
        new_states = trajectories.env.States(new_states)
        return Trajectories(
            env=trajectories.env,
            states=new_states,
            actions=new_actions,
            log_probs=trajectories.log_probs,
            when_is_done=new_when_is_done,
            is_backward=False,
        )

    def to_transitions(self) -> Transitions:
        """Returns a `Transitions` object from the trajectories"""
        raise NotImplementedError()

    def to_states(self) -> States:
        """Returns a `States` object from the trajectories, containing all states in the trajectories"""
        states = self.states.flatten()
        return states[~states.is_sink_state]

    def to_non_initial_intermediary_and_terminating_states(
        self,
    ) -> tuple[States, States]:
        """Returns a tuple of `States` objects from the trajectories, containing all non-initial intermediary and all terminating states in the trajectories

        Returns:
            Tuple[States, States]: - All the intermediary states in the trajectories that are not s0.
                                   - All the terminating states in the trajectories that are not s0.
        """
        states = self.states
        intermediary_states = states[~states.is_sink_state & ~states.is_initial_state]
        terminating_states = self.last_states
        terminating_states.log_rewards = self.log_rewards
        return intermediary_states, terminating_states
