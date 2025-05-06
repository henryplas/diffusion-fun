# traj_diffusion.py

import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from cleandiffuser.diffusion import ContinuousDiffusionSDE
from cleandiffuser.nn_diffusion import PearceMlp
from cleandiffuser.nn_condition import PearceObsCondition
from cleandiffuser.utils import loop_dataloader


def main():
    # 1) Load cleaned trajectories (.npz of pure arrays)
    data = np.load("drone_trajectories_clean.npz")
    obs_arr = data["observations"]   # (N, T, obs_dim)
    acts_arr = data["actions"]        # (N, T, act_dim)

    # flatten episodes × timesteps for single-step DBC
    N, T, obs_dim = obs_arr.shape
    _, _, act_dim = acts_arr.shape
    obs_flat = obs_arr.reshape(-1, obs_dim)
    acts_flat = acts_arr.reshape(-1, act_dim)

    # 2) Build PyTorch Dataset & DataLoader
    dataset = TensorDataset(
        torch.from_numpy(obs_flat).float(),
        torch.from_numpy(acts_flat).float()
    )
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )

    # 3) Build diffusion policy components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_diffusion = PearceMlp(
        act_dim=act_dim, To=1, emb_dim=128, hidden_dim=512,
        timestep_emb_type="untrainable_fourier"
    ).to(device)
    nn_condition = PearceObsCondition(
        obs_dim=obs_dim, emb_dim=128, flatten=True, dropout=0.0
    ).to(device)
    actor = ContinuousDiffusionSDE(
        nn_diffusion, nn_condition,
        x_max=torch.ones(act_dim, device=device),
        x_min=-torch.ones(act_dim, device=device),
        ema_rate=0.9999, device=device
    )

    # 4) Training loop (DBC imitation)
    n_steps = 5000
    log_interval = 1_000
    save_interval = 5000
    save_dir = "drone_diffuser_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    actor.train()
    total_loss = 0.0
    for step, (obs_batch, act_batch) in enumerate(loop_dataloader(loader), start=1):
        obs_batch = obs_batch.to(device)
        act_batch = act_batch.to(device)

        out = actor.update(x0=act_batch, condition=obs_batch)
        # out['loss'] is a float; no .item() needed
        total_loss += out["loss"]

        if step % log_interval == 0:
            print(f"[{step}/{n_steps}] loss={total_loss/log_interval:.4f}")
            total_loss = 0.0

        if step % save_interval == 0:
            ckpt = os.path.join(save_dir, f"drone_diff_{step//1000}k.pt")
            actor.save(ckpt)
            print(f"Saved checkpoint: {ckpt}")

        if step >= n_steps:
            break

    # final save
    actor.save(os.path.join(save_dir, "drone_diff_final.pt"))
    print("Training complete.")


if __name__ == "__main__":
    from torch.multiprocessing import freeze_support
    freeze_support()
    main()

