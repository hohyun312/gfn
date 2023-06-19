import numpy as np
import copy


class Buffer:
    def __init__(self, size=1000):
        self.size = size
        self.queue = np.zeros(size, dtype=object)
        self._pos = 0
        self._nonzero = 0

        self._states = []
        self._actions = []

    def put(self, state, action=None, reward=None):
        self._states.append(copy.copy(state))
        self._actions.append(copy.copy(action))

        if action is None:
            self._actions.pop()
            states = np.array(self._states)
            actions = np.array(self._actions)
            self.queue[self._pos] = (states, actions, reward)
            self._pos += 1
            self._pos = self._pos % self.size
            self._nonzero = min(self._nonzero + 1, self.size)
            self._states = []
            self._actions = []

    def sample(self, size=5):
        idx = np.random.choice(range(self._nonzero), size=size, replace=True)
        samples = self.queue[idx]
        states = np.stack(
            [s for s, a, r in samples], axis=0
        )  # (batch, trajectory_length, feature_dim)
        actions = np.array([a for s, a, r in samples])
        rewards = np.array([r for s, a, r in samples])
        return states, actions, rewards
