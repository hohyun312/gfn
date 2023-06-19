import itertools
from collections import defaultdict, deque

import networkx as nx
import numpy as np


class BitVecEnv:
    def __init__(self, ndim=5, reward_fn=None):
        """Bit vector environment where states are bit vector.
        The states are represented as 1-d array of length `ndim`.
        The initial state is represented as zero array. E.g. [0, 0, 0, 0] for `ndim = 4`.
        The action is to convert a single dimension of the state that is currently 0 into 1.
        Args:
            ndim (int, optional): dimension of the state. Defaults to 5.
            reward_fn (Callable, optional): a reward function that takes `ndim` state
                array as an input and outputs a scalar reward when terminal state is reached.
                For non-terminal states, it has to output `None` value. If `None` is provided,
                it defaults to reward 1's at higher indices when `ndim - 1` step is reached.
                See `default_reward_fn` for implementation.
        """
        self.ndim = ndim
        self.reward_fn = reward_fn

        if self.reward_fn is None:
            self.reward_fn = self.default_reward_fn

        self._cur_state = None
        self.G, self.pos = self._compute_state_graph_and_postion()

    def default_reward_fn(self, state):
        assert state.ndim == 1
        if state.sum() < self.ndim - 1:
            return None
        else:
            return state @ np.arange(len(state))

    def reset(self):
        self._cur_state = np.zeros(self.ndim)
        return self._cur_state

    def step(self, action):
        assert self._cur_state[action] == 0
        self._cur_state[action] = 1
        reward = self.reward_fn(self._cur_state)
        if reward is None:  # non-terminal
            reward = 0
            done = False
        else:
            done = True
        return self._cur_state, reward, done

    def feasible_actions(self):
        return (1 - self._cur_state).nonzero()[0]

    def _make_full_graph(self):
        states = list(itertools.product((0, 1), repeat=self.ndim))
        adj = defaultdict(list)
        for s in states:
            for i, b in enumerate(s):
                if b == 0:
                    s_ = s[:i] + (1,) + s[i + 1 :]
                    adj[s] += [s_]
        G = nx.DiGraph()
        G.add_edges_from([(k, i) for k, v in adj.items() for i in v])
        return G

    def _compute_state_graph_and_postion(self):
        G = self._make_full_graph()

        # compute rewards for each state
        for node, d in G.nodes(data=True):
            rew = self.reward_fn(np.array(node))
            d["non_terminal"] = True if rew is None else False
            d["reward"] = 0 if rew is None else rew

        # BFS
        s0 = (0,) * self.ndim
        frontier = deque([s0])
        pos = {s0: [0, 0]}
        depth2height = defaultdict(int)
        depth2height[0] = 1
        while frontier:
            cur = frontier.popleft()
            new_nodes = sorted(G.adj[cur].keys(), reverse=True)
            for n in new_nodes:
                if n not in pos:
                    d = pos[cur][0] + 1
                    pos[n] = [d, depth2height[d]]
                    depth2height[d] += 1
                    if G.nodes[n]["non_terminal"]:
                        frontier.append(n)

        # remove out-edges from terminal states
        for n, d in G.nodes(data=True):
            if not d["non_terminal"]:  # if terminal
                edges_lst = [(n, v) for v in G.adj[n].keys()]
                G.remove_edges_from(edges_lst)

        # remove unvisited states
        G_sub = G.subgraph(pos)

        # recompute position for pretty visualization
        max_height = max(depth2height.values())
        for p in pos:
            d, h = pos[p]
            gap = max_height / (depth2height[d] + 1)
            pos[p][1] = h * gap + gap

        return G_sub, pos

    def get_all_states(self):
        return np.array(self.G.nodes)

    def visualize(
        self,
        font_size=6,
        labels=True,
        color_terminals=True,
        action_probs=None,
        logZ=None,
    ):
        """Visualize environment.
        Args:
            action_probs (array, optional): if loZ is set to None, edge width will be set to action probabilities.
                `action_probs` has shape (number of states, number of actions)
            logZ (Callable, optional): if provided with action_probs, edge width will be set to edge flow.
        """
        kargs = dict(
            G=self.G,
            pos=self.pos,
            font_size=font_size,
            node_color=["#aaaaaa"] * self.G.number_of_nodes(),
        )

        if labels:
            labels = {s: str(s).strip("()").replace(", ", "") for s in self.G.nodes}
            kargs.update({"labels": labels})

        if color_terminals:
            node_color = [
                "#aaaaaa" if d["non_terminal"] else "#0fafaf"
                for n, d in self.G.nodes(data=True)
            ]
            kargs.update({"node_color": node_color})

        if action_probs is not None:  # compute edge width
            state_flow = defaultdict(float)
            if logZ is not None:
                s0 = (0,) * self.ndim
                state_flow[s0] = np.exp(logZ)

            edge2id = dict(zip(self.G.edges, range(self.G.number_of_edges())))
            width = np.zeros(self.G.number_of_edges())
            for i, node in enumerate(sorted(self.G.nodes)):
                for nei in self.G.adj[node]:
                    act = np.argmax(np.array(nei) - np.array(node))

                    if logZ is None:
                        width[edge2id[(node, nei)]] = action_probs[i, act]
                    else:
                        edge_flow = state_flow[node] * action_probs[i, act]
                        width[edge2id[(node, nei)]] = edge_flow
                        state_flow[nei] += edge_flow
            kargs.update({"width": width})

        return nx.draw_networkx(with_labels=True, **kargs)


if __name__ == "__main__":

    def reward_fn(state):
        assert state.ndim == 1
        if state[:3].sum() == 3:
            return 1

    env = BitVecEnv(ndim=5)
    env.visualize()
