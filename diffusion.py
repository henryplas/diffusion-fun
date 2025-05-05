import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1) Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Diffusion hyperparameters
T = 100     # total noising steps
betas = torch.linspace(1e-4, 0.02, T).to(device)  
alphas = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0)  # ∏_{s=1}^t α_s

# 3) Noise-predicting network (MLP)
class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x, t):
        # x: [batch,1], t: [batch] (ints in [0,T))
        t_embed = (t.float() / T).unsqueeze(1)       # simple time embed
        inp     = torch.cat([x, t_embed], dim=1)    
        return self.net(inp)


# 4) Training loop
batch_size = 128

model     = DiffusionModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
mse       = nn.MSELoss()

# sample clean data x0 from Mixture of Gaussians
# x0 is the original distribution that is trained on,  can be three different distributions hypothetically
comp  = torch.randint(0, 2, (batch_size,), device=device)
means = torch.where(comp==0, -2.0, 2.0)
x0    = (means + 0.5*torch.randn(batch_size, device=device)).unsqueeze(1)


for step in range(10_000):

    # pick random timestep t
    t = torch.randint(0, T, (batch_size,), device=device)
    α_bar_t = alphas_bar[t].unsqueeze(1)

    # forward noising: x_t = √α̅_t·x0 + √(1−α̅_t)·ε
    noise = torch.randn_like(x0)
    xt    = torch.sqrt(α_bar_t)*x0 + torch.sqrt(1-α_bar_t)*noise

    # predict ε and step optimizer
    pred = model(xt, t)
    loss = mse(pred, noise)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    if step % 200 == 0:
        print(f"[{step:4d}/1000] loss = {loss.item():.4f}")

# 5) Sampling: start from pure noise and apply learned reverse chain
with torch.no_grad():
    x = torch.randn(batch_size, 1, device=device)
    for t_inv in reversed(range(T)):
        t_batch = torch.full((batch_size,), t_inv, device=device, dtype=torch.long)
        ε_pred  = model(x, t_batch)
        β       = betas[t_inv]
        α       = alphas[t_inv]
        α_bar   = alphas_bar[t_inv]

        # posterior mean: (1/√α)( x - (β/√(1−α̅)) ε_pred )
        mean = (1/torch.sqrt(α)) * (x - (β/torch.sqrt(1-α_bar))*ε_pred)

        # add noise except at final step
        noise = torch.randn_like(x) if t_inv > 0 else torch.zeros_like(x)
        x     = mean + torch.sqrt(β)*noise

    samples = x.cpu().numpy().flatten()

# 6) Visualize the generated 1-D samples
plt.hist(samples, bins=50)
plt.title("1D Diffusion Model Samples")
plt.xlabel("x")
plt.ylabel("Frequency")
plt.show()