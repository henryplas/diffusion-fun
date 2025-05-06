# drone_diffusion_test.py

import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath("external/drone_rl"))
from drone import DroneGymEnv
from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import PearceMlp
from cleandiffuser.nn_condition import PearceObsCondition

def build_actor(act_dim, obs_dim, checkpoint_path, device):
    # 1) Re-create networks (must match your training setup)
    nn_diffusion = PearceMlp(
        act_dim=act_dim, To=1, emb_dim=128, hidden_dim=512,
        timestep_emb_type="untrainable_fourier"
    ).to(device)
    nn_condition = PearceObsCondition(
        obs_dim=obs_dim, emb_dim=128, flatten=True, dropout=0.0
    ).to(device)

    # 2) Wrap into the SDE‐based diffusion policy
    actor = ContinuousDiffusionSDE(
        nn_diffusion, nn_condition,
        x_max=torch.ones(act_dim, device=device),
        x_min=-torch.ones(act_dim, device=device),
        ema_rate=0.9999,
        device=device
    )

    # 3) Load the trained weights
    actor.load(checkpoint_path)   # e.g. "drone_diffuser_checkpoints/drone_diff_final.pt"
    actor.eval()
    return actor

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1) Instantiate environment and actor ---
    env = DroneGymEnv()
    obs = env.reset()
    obs_dim = obs.shape[-1]      # your env’s observation dimension
    act_dim = env.action_space.shape[-1]

    checkpoint = os.path.join("drone_diffuser_checkpoints", "drone_diff_final.pt")
    actor = build_actor(act_dim, obs_dim, checkpoint, device)

    # --- 2) Run evaluation episodes ---
    num_episodes = 50
    max_steps = 100
    returns = []

    for ep in range(1, num_episodes+1):
        obs = env.reset()
        ep_ret = 0.0
        done = False

        # zero‐prior conditions (shape: [batch=1, act_dim])
        prior = torch.zeros((1, act_dim), device=device)

        for t in range(max_steps):
            # sample an action given the current state
            obs_tensor = torch.tensor(obs[None], device=device, dtype=torch.float32)  # [1, obs_dim]
            acts, _ = actor.sample(
                prior=prior,
                solver="ddpm",             # or another solver
                n_samples=1,
                sample_steps=20,           # fewer steps for speed; increase for fidelity
                condition_cfg=obs_tensor,
                w_cfg=1.0
            )
            action = acts.cpu().numpy()[0]  # shape (act_dim,)

            # step in the env
            obs, rew, done, info = env.step(action)
            ep_ret += rew
            if done:
                break

        returns.append(ep_ret)
        print(f"Episode {ep:2d} return: {ep_ret:.2f}")

    # --- 3) Summarize ---
    returns = np.array(returns)
    print(f"\nEvaluation over {num_episodes} episodes:")
    print(f"  Mean return: {returns.mean():.2f}")
    print(f"  Std return : {returns.std():.2f}")

if __name__ == "__main__":
    # on Windows, guard multiprocessing
    from torch.multiprocessing import freeze_support
    freeze_support()
    main()
