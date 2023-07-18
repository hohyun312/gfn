import torch
from tqdm import tqdm

from src.envs.bitflip import BitFlipEnv
from gfn import LogitPBEstimator, LogitPFEstimator, LogZEstimator
from gfn.losses import TBParametrization, TrajectoryBalance
from gfn.samplers import DiscreteActionsSampler, TrajectoriesSampler


def print_evaluate():
    from collections import Counter
    import numpy as np

    traj = trajectories_sampler.sample(500)
    shape = (1, 1, traj.states.states_tensor.shape[-1])
    index = traj.when_is_done[None, :, None].repeat(shape)
    final_states = traj.states.states_tensor.gather(dim=0, index=index - 1).squeeze(0)
    final_states_tuples = [tuple(x.cpu().numpy()) for x in final_states]

    counter = Counter()
    counter.update({tuple(x): 0 for x in env.all_states.states_tensor.numpy()})
    counter.update(final_states_tuples)
    counts = np.array(list(counter.values()))
    empirical_freq = counts / counts.sum()

    rewards = env.log_reward(env.all_states).exp().numpy()
    theoretical_freq = rewards / rewards.sum()

    print("Empirical Frequency  :", empirical_freq.round(2))
    print("Theoretical Frequency:", theoretical_freq.round(2))

    print(
        "Empirical L1 error   :",
        np.linalg.norm(empirical_freq - theoretical_freq, ord=1),
    )
    print("Learned     log Z:", parametrization.logZ)
    print("Theoretical log Z:", np.log(np.sum(rewards)))


if __name__ == "__main__":
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    env = BitFlipEnv(length=4, device_str=device_str)

    logit_PF = LogitPFEstimator(env=env, module_name="NeuralNet")
    logit_PB = LogitPBEstimator(
        env=env,
        module_name="NeuralNet",
        torso=logit_PF.module.torso,  # To share parameters between PF and PB
    )
    logZ = LogZEstimator(torch.tensor(0.0))

    parametrization = TBParametrization(logit_PF, logit_PB, logZ)

    actions_sampler = DiscreteActionsSampler(estimator=logit_PF)
    trajectories_sampler = TrajectoriesSampler(env=env, actions_sampler=actions_sampler)

    loss_fn = TrajectoryBalance(parametrization=parametrization)

    def unique_params(params):
        return list(set(params))

    params = [
        {
            "params": unique_params(
                [
                    val
                    for key, val in parametrization.parameters.items()
                    if "logZ" not in key
                ]
            ),
            "lr": 0.001,
        },
        {
            "params": [
                val for key, val in parametrization.parameters.items() if "logZ" in key
            ],
            "lr": 0.1,
        },
    ]
    optimizer = torch.optim.Adam(params)

    for i in (pbar := tqdm(range(1000))):
        trajectories = trajectories_sampler.sample(n_trajectories=16)
        optimizer.zero_grad()
        loss = loss_fn(trajectories)
        loss.backward()
        optimizer.step()
        if i % 25 == 0:
            pbar.set_postfix({"loss": loss.item()})

    print_evaluate()
