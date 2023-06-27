import random
import numpy as np
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.distributions import Categorical


from src.model import TBModel
from src.buffer import Buffer
from src.bitvec import BitVecEnv


def train(
    env, model, buffer, optimizer, device, n_episodes, batch_size, update_start, epsilon
):
    tb_losses = []
    logZs = []

    for episode in tqdm(range(n_episodes), desc="training"):
        state = env.reset()

        done = False
        while not done:
            if random.random() < epsilon:
                action = model.act(state[None, :]).item()
            else:
                action = np.random.choice(env.feasible_actions())

            buffer.put(state, action)
            state, reward, done = env.step(action)

        buffer.put(state, None, reward)

        if episode >= update_start - 1:
            states, actions, rewards = buffer.sample(batch_size)
            acts = torch.tensor(actions).to(device)
            rewards = torch.tensor(rewards).to(device)

            P_F, P_B = model(states)

            logP_F = Categorical(logits=P_F[:, :-1, :]).log_prob(acts)
            logP_B = Categorical(logits=P_B[:, 1:, :]).log_prob(acts)

            total_logP_F = logP_F.sum(dim=1)
            total_logP_B = logP_B.sum(dim=1)
            logR = rewards.clamp(min=1e-20).log()

            loss = (model.logZ + total_logP_F - logR - total_logP_B).pow(2).mean()

            tb_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            logZs.append(model.logZ.item())

    return tb_losses, logZs


def print_eval(model, env, n_iters=200):
    end_states = []
    for episode in tqdm(range(n_iters), desc="evaluating"):
        state = env.reset()

        done = False
        while not done:
            action = model.act(state[None, :])
            state, _, done = env.step(action)

        end_states += [tuple(state)]

    counter = Counter(end_states)

    rew = np.array(
        [
            d["reward"]
            for node, d in sorted(env.G.nodes(data=True))
            if not d["non_terminal"]
        ]
    )
    rew_dist = rew / rew.sum() 
    print("Reward distribution   :", rew_dist.round(3))

    k = sorted(counter)
    bins = np.array([counter[i] for i in k])
    emp_dist = bins / bins.sum()
    print("Policy visit frequency:", emp_dist.round(3))

    print("Empirical L1 error:", np.linalg.norm(rew_dist - emp_dist, ord=1))
    print("Learned log Z :", model.logZ.item())
    print("Sum of Rewards:", np.log(rew.sum()))


if __name__ == "__main__":
    # define model and environment
    ndim = 5
    buffer_size = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TBModel(ndim=ndim, hidden_dims=[20, 20]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    buffer = Buffer(size=buffer_size)
    env = BitVecEnv(ndim=ndim)

    # train
    n_episodes = 2000
    batch_size = 32
    update_start = 100
    epsilon = 0.95
    tb_losses, logZs = train(
        env,
        model,
        buffer,
        optimizer,
        device,
        n_episodes,
        batch_size,
        update_start,
        epsilon,
    )

    # eval
    print_eval(model, env, n_iters=200)

    # viz1: action probability
    with torch.no_grad():
        logits, _ = model(np.array(env.G.nodes))
        action_probs = logits.softmax(dim=1).cpu().numpy()

    env.visualize(font_size=8, action_probs=action_probs)
    plt.show()

    # viz2: training loss & logZ
    plt.plot(logZs, label="logZ")
    plt.plot(tb_losses, label="loss")
    plt.legend()
    plt.show()
