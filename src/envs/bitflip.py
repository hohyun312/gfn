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


class BitFlipEnv(Env):
    def __init__(self, length: int = 5, device_str: Literal["cpu", "cuda"] = "cpu"):
        """Bit-flipping environment where states are binary vector of `length`
        and any action applied to a state flips a single bit.
        The initial state is represented as zero array. E.g. [0, 0, 0, 0] for `length = 4`.
        The action is to convert a single dimension of the state that is currently 0 into 1.
        Args:
            length (int, optional): dimension of the state. Defaults to 5.
        """
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
        all_states_list = list(itertools.product((0, 1), repeat=self.length))
        all_states_tensor = torch.LongTensor(all_states_list)
        return self.States(all_states_tensor)


import itertools
from collections import defaultdict, deque

import networkx as nx
import numpy as np


class Visualizer:
    def __init__(self, length, reward_fn=None):
        self.length = length
        self.reward_fn = reward_fn
        self.G, self.pos = self._compute_state_graph_and_postion()

    def _make_full_graph(self):
        states = list(itertools.product((0, 1), repeat=self.length))
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
            d["non_terminal"] = True if rew <= 1e-8 else False
            d["reward"] = rew

        # BFS
        s0 = (0,) * self.length
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
                    frontier.append(n)

        # remove out-edges from terminal states
        # for n, d in G.nodes(data=True):
        #     if not d["non_terminal"]:  # if terminal
        #         edges_lst = [(n, v) for v in G.adj[n].keys()]
        #         G.remove_edges_from(edges_lst)

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
        node_flow=False,
        node_reward=False,
        node_size=100,
        **kargs
    ):
        """Visualize environment.
        Args:
            action_probs (array, optional): if logZ is set to None, edge width will be set to action probabilities.
                `action_probs` has shape (number of states, number of actions)
            logZ (float, optional): if provided with action_probs, edge width will be set to edge flow.
            state_reward: if True, rewards given to states will be drawn with node edge.
            node_size: controls relative sizes of the nodes.
        """
        kargs = dict(
            G=self.G,
            pos=self.pos,
            font_size=font_size,
            node_color=["#aaaaaa"] * self.G.number_of_nodes(),
            **kargs
        )

        if labels:
            labels = {s: str(s).strip("()").replace(", ", "") for s in self.G.nodes}
            kargs.update({"labels": labels})

        if color_terminals:
            node_color = [
                "#aaaaaa" if d["non_terminal"] else "#eb928f"
                for n, d in self.G.nodes(data=True)
            ]
            kargs.update({"node_color": node_color})

        if action_probs is not None or node_flow:  # compute edge width
            state_flow = defaultdict(float)
            if logZ is not None:
                s0 = (0,) * self.length
                state_flow[s0] = np.exp(logZ)

            edge2id = dict(zip(self.G.edges, range(self.G.number_of_edges())))
            width = np.zeros(self.G.number_of_edges())
            node_weight = np.zeros(self.G.number_of_nodes())
            for i, node in enumerate(sorted(self.G.nodes)):
                for nei in self.G.adj[node]:
                    act = np.argmax(np.array(nei) - np.array(node))

                    if logZ is None:
                        width[edge2id[(node, nei)]] = action_probs[i, act]
                    else:
                        edge_flow = state_flow[node] * action_probs[i, act]
                        width[edge2id[(node, nei)]] = edge_flow
                        state_flow[nei] += edge_flow
                node_weight[i] = state_flow[node] * action_probs[i, -1]

            if action_probs is not None:
                kargs.update({"width": width})

            if node_flow:
                kargs.update({"node_size": node_weight * node_size})

        nx.draw_networkx(with_labels=True, **kargs)

        if node_reward:
            rewards = np.array([v["reward"] for _, v in self.G.nodes(data=True)])
            nodes = nx.draw_networkx_nodes(self.G, self.pos)
            nodes.set_color("#00000000")
            nodes.set_edgecolor("#000000")
            nodes.set_sizes(rewards * node_size)

    def get_action_probs(self, env, model):
        State = env.make_States_class()
        with torch.no_grad():
            state = State(torch.tensor(self.get_all_states()))
            logits = model(state).cpu().numpy()
            mask = self.get_all_states().astype(bool)
            logits[:, :-1][mask] = -float("inf")
            action_probs = softmax(logits, axis=1)

        return action_probs

    # def get_flow_from_probs(self, action_probs, Z):
    #     flows_dict = {tuple(s): 0 for s in self.get_all_states()}
    #     probs_dict = {tuple(s): p for s, p in zip(self.get_all_states(), action_probs)}

    #     # BFS
    #     s0 = (0,) * self.length
    #     frontier = deque([s0])
    #     flows_dict[s0] = Z
    #     visited = {s0}
    #     while frontier:
    #         cur = frontier.popleft()
    #         new_nodes = sorted(self.G.adj[cur].keys(), reverse=True)
    #         for n in new_nodes:
    #             # find action index
    #             for act, (old, new) in enumerate(zip(cur, n)):
    #                 if old < new:
    #                     break

    #             flows_dict[n] = flows_dict[cur] * probs_dict[cur][act]

    #             if n not in visited:
    #                 visited.add(n)
    #                 frontier.append(n)

    #     edge_flow = np.zeros_like(action_probs)
    #     for i, arr in enumerate(self.get_all_states()):
    #         k = tuple(arr)
    #         edge_flow[i] = flows_dict[k] * probs_dict[k]

    #     return edge_flow


def softmax(x, axis=0):
    z = np.exp(x - np.max(x))
    return z / z.sum(axis=axis, keepdims=True)
