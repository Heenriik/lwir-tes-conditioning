import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


# --- Torch-side Planck helpers (used by the physics-consistency loss term) ---
# Constants match src/forward_model/planck.py so the loss term agrees with the
# analytical Jacobian and the numpy forward model used elsewhere.
#
# We pre-combine the small constants into the standard radiation constants:
#   c1 = 2·h·c²     ≈ 1.19e-16  W·m²/sr
#   c2 = h·c / k    ≈ 0.0144    m·K   (second radiation constant)
# This is essential for float32 autograd: the naive form (h·c)/(λ·k·T)
# would underflow during the backward pass because λ·k·T ≈ 1e-26 is fine in
# forward but its square (needed for ∂/∂T of a/b) is ≈ 1e-51, well below the
# float32 min subnormal (~1.4e-45), producing NaN gradients.
_H = 6.626e-34
_C = 3.0e8
_K = 1.381e-23
_C1 = 2.0 * _H * _C * _C    # 1.19e-16
_C2 = _H * _C / _K          # 0.01439


def _planck_torch(wl, T):
    """
    Planck radiance, torch version. Computed in float64 internally because
    autograd through the float32 form produces inf/NaN gradients via a
    subnormal-underflow in the chain rule:

        ∂B/∂T involves dividing by (λ⁵·(exp(x)-1))² ≈ (1e-25)² = 1e-50,
        which falls below float32 min subnormal (~1.4e-45) and saturates to 0,
        so the gradient becomes inf and propagates as NaN.

    Float64 handles 1e-50 trivially (min normal ≈ 2e-308). The output is
    cast back to float32 to match the rest of the network's dtype.

    Args:
        wl (Tensor): wavelengths in metres, shape (N,)
        T (Tensor): temperatures in Kelvin, shape (M,)
    Returns:
        Tensor: radiance W/m²/sr/m, shape (M, N), float32
    """
    wl_b = wl.double().unsqueeze(0)  # (1, N) float64
    T_b = T.double().unsqueeze(1)    # (M, 1) float64
    x = _C2 / (wl_b * T_b)
    B = _C1 / (wl_b**5 * (torch.exp(x) - 1.0))
    return B.float()


def _compute_radiance_torch(wl, T, eps, L_env):
    """
    L = ε·B(T) + (1-ε)·L_env, torch version.

    Args:
        wl (Tensor): (N,)
        T (Tensor): (M,)
        eps (Tensor): (M, N)
        L_env (Tensor): (N,) or (M, N)
    Returns:
        Tensor: (M, N)
    """
    B = _planck_torch(wl, T)
    L_env_b = L_env.unsqueeze(0) if L_env.ndim == 1 else L_env
    return eps * B + (1.0 - eps) * L_env_b


class ResidualMLP(nn.Module):
    """
    Lipschitz-constrained Multilayer Perceptron (MLP) residual corrector.

    Maps measured radiance L (N,) to a parameter-space correction Δx (N+1,)
    where x = [T, ε_1,...,ε_N].

    Spectral normalization (torch.nn.utils.spectral_norm) is applied to every
    linear layer when use_spectral_norm=True. This enforces σ_max(W_k) ≤ 1
    per layer, so the global Lipschitz constant is bounded above by the product
    of per-layer spectral norms (exposed via lipschitz_upper_bound).
    """

    def __init__(self, input_dim, output_dim, hidden_dims=None, use_spectral_norm=True):
        """
        Args:
            input_dim (int): number of spectral bands N
            output_dim (int): N+1 (one correction per parameter T + ε_i)
            hidden_dims (list[int] or None): hidden layer widths; defaults to [64, 64]
            use_spectral_norm (bool): wrap layers with spectral normalization
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            if use_spectral_norm:
                linear = spectral_norm(linear)
            layers.append(linear)
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU(0.01))
        self.net = nn.Sequential(*layers)
        self.use_spectral_norm = use_spectral_norm
        # L_scale normalises raw radiance to O(1) before the first linear layer.
        # LWIR radiance is ~1e7 W/m²/sr/m; without scaling, spectral-norm layers
        # (per-layer gain ≤ 1) cannot bridge to O(1) parameter corrections.
        # Set by train_residual_model from the training data.
        self.register_buffer("L_scale", torch.tensor(1.0))

    def forward(self, x):
        return self.net(x / self.L_scale)

    @property
    def lipschitz_upper_bound(self):
        """
        Upper bound on Lipschitz constant: product of per-layer spectral norms.

        Only meaningful when use_spectral_norm=True and after at least one
        forward pass (spectral_norm requires a forward pass to initialise u/v).
        """
        bound = 1.0
        for module in self.net:
            if isinstance(module, nn.Linear):
                W = module.weight.data
                sigma = torch.linalg.norm(W, ord=2).item()
                bound *= sigma
        return bound


# Backwards-compatible alias for the original 2-layer skeleton
ResidualNet = ResidualMLP


def train_residual_model(model, L_train, true_params_batch, physics_params_batch,
                          n_epochs=200, lr=1e-3, weight_decay=1e-4, device=None,
                          lambda_phys=0.0, wavelengths=None, L_env=None):
    """
    Train the residual MLP to correct physics inversion errors.

    Target: true_params ≈ physics_params + model(L)

    Parameters are normalised before training:
        T / T_scale  (T_scale = 300 K)
        ε as-is (already in [0,1])

    Optimiser: AdamW (Loshchilov & Hutter 2019, "Decoupled Weight Decay
    Regularization", arxiv.org/pdf/1711.05101) — decouples L2 weight decay from the adaptive
    moment update, giving cleaner hyperparameter settings than vanilla Adam.

    Physical-consistency loss (SENSE Eq. 10, term λ₁):
        If lambda_phys > 0, the loss adds  λ_phys · ‖P(x̂) - L‖² / (L_scale)²,
        matching SENSE Eq. 10 (their `y` is our `L`).
        P(x̂) = ε̂·B(T̂) + (1-ε̂)·L_env is the forward radiance. T̂/ε̂ are
        clamped to physical bounds before P() is evaluated (random init can
        push the unconstrained correction far outside (200, 500) K, which
        overflows _planck_torch). The optimiser also runs gradient clipping
        as a safety net against the phys term's nonlinear amplification.

        Magnitudes: ‖P(x̂) - L‖² / L_scale² has natural range O(1e-10) at
        the TRF solution (underdetermined physics fits noise exactly) up to
        ~1e-3 when the network moves x̂ off-TRF. Accuracy loss is ~5e-2.
        For meaningful regularisation strength, sweep lambda_phys ∈
        [10, 100, 1000, 10000, 100000].

    Args:
        model (ResidualMLP): untrained or partially trained model
        L_train (np.ndarray): measured radiance, shape (M, N)
        true_params_batch (np.ndarray): [T, ε_1,...,ε_N], shape (M, N+1)
        physics_params_batch (np.ndarray): physics inversion output, shape (M, N+1)
        n_epochs (int): training epochs
        lr (float): AdamW learning rate
        weight_decay (float): AdamW weight decay (L2 regularisation)
        device (str or None): 'cpu', 'cuda', or None (auto-detect)
        lambda_phys (float): weight for physical-consistency term (0 disables)
        wavelengths (np.ndarray or None): metres, shape (N,) — required if lambda_phys>0
        L_env (np.ndarray or None): downwelling radiance, shape (N,) — required if lambda_phys>0

    Returns:
        list[float]: loss per epoch (combined acc + phys when lambda_phys>0)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    L_scale = float(np.abs(L_train).max())
    if L_scale < 1e-30:
        L_scale = 1.0
    model.L_scale.fill_(L_scale)

    T_scale = 300.0
    def _normalise(params):
        p = params.copy().astype(np.float32)
        p[:, 0] /= T_scale
        return p

    L_t = torch.tensor(L_train.astype(np.float32), device=device)
    true_t = torch.tensor(_normalise(true_params_batch), device=device)
    phys_t = torch.tensor(_normalise(physics_params_batch), device=device)

    # Warm up spectral_norm: first forward uses random u/v vectors to estimate
    # σ_max(W). A poor estimate gives a poor W normalisation, and the resulting
    # output can have extreme values whose gradients through _planck_torch
    # become NaN. A few no-grad forward passes let the power iteration converge
    # before any backward pass sees the weights.
    with torch.no_grad():
        for _ in range(5):
            _ = model(L_t[:32])

    use_phys = lambda_phys > 0.0
    if use_phys:
        if wavelengths is None or L_env is None:
            raise ValueError("lambda_phys > 0 requires `wavelengths` and `L_env`")
        wl_t = torch.tensor(np.asarray(wavelengths, dtype=np.float32), device=device)
        L_env_t = torch.tensor(np.asarray(L_env, dtype=np.float32), device=device)
        with torch.no_grad():
            phys_T0 = phys_t[:, 0] * T_scale
            phys_eps0 = phys_t[:, 1:]
            L_pred0 = _compute_radiance_torch(wl_t, phys_T0, phys_eps0, L_env_t)
            phys_loss0 = ((L_pred0 - L_t) ** 2 / L_scale**2).mean()
            acc_loss0 = nn.MSELoss()(phys_t, true_t)
        # Anchor phys_scale to acc_loss_init magnitude so lambda_phys=1 means
        # "phys term initially comparable to acc term"; floor avoids the
        # underdetermined-fit zero-residual division blow-up.
        phys_scale = float(acc_loss0.item()) + 1e-30
        print(f"  [phys] init: acc_loss={float(acc_loss0):.4e}"
              f"  phys_loss={float(phys_loss0):.4e}"
              f"  phys_scale={phys_scale:.4e}  lambda_phys={lambda_phys:g}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = []

    n_skipped = 0
    first_skip_msg = None

    model.train()
    for ep in range(n_epochs):
        optimiser.zero_grad()
        correction = model(L_t)
        pred = phys_t + correction
        acc_loss = loss_fn(pred, true_t)

        if use_phys:
            # Clamp pred to physical bounds before evaluating P() — random init
            # can push corrections far outside (200, 500) K, which overflows the
            # exp() in _planck_torch.
            T_pred = torch.clamp(pred[:, 0] * T_scale, 200.0, 500.0)
            eps_pred = torch.clamp(pred[:, 1:], 0.01, 1.0)
            L_pred = _compute_radiance_torch(wl_t, T_pred, eps_pred, L_env_t)
            phys_loss = ((L_pred - L_t) ** 2 / L_scale**2).mean()
            loss = acc_loss + lambda_phys * (phys_loss / phys_scale)
        else:
            loss = acc_loss

        # NaN/Inf guard: skip the step rather than corrupting weights.
        if not torch.isfinite(loss):
            if first_skip_msg is None:
                first_skip_msg = (f"non-finite loss at epoch {ep}"
                                   f" (acc={float(acc_loss):.3e},"
                                   f" phys={float(phys_loss) if use_phys else 0:.3e})")
            optimiser.zero_grad()
            history.append(float("nan"))
            n_skipped += 1
            continue

        loss.backward()
        if use_phys:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        bad_grad = any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in model.parameters())
        if bad_grad:
            if first_skip_msg is None:
                first_skip_msg = f"non-finite gradient at epoch {ep}"
            optimiser.zero_grad()
            history.append(float("nan"))
            n_skipped += 1
            continue
        optimiser.step()
        history.append(loss.item())

    if n_skipped > 0:
        print(f"  warning: skipped {n_skipped}/{n_epochs} epochs"
              f" (first reason: {first_skip_msg})")

    model.eval()
    return history


def compute_residual_jacobian(model, L_input):
    """
    Compute J_residual = ∂r_θ/∂L via PyTorch autograd.

    Args:
        model (ResidualMLP): trained model (eval mode)
        L_input (np.ndarray or torch.Tensor): shape (N,) or (M, N)

    Returns:
        np.ndarray: shape (N+1, N) for single input, or (M, N+1, N) for batch
    """
    model.eval()
    if isinstance(L_input, np.ndarray):
        L_tensor = torch.tensor(L_input.astype(np.float32))
    else:
        L_tensor = L_input.float()

    batched = L_tensor.ndim == 2
    if not batched:
        L_tensor = L_tensor.unsqueeze(0)

    jacs = []
    for i in range(L_tensor.shape[0]):
        x = L_tensor[i].detach().requires_grad_(True)
        J = torch.autograd.functional.jacobian(model, x)
        jacs.append(J.detach().numpy())

    result = np.stack(jacs, axis=0)  # (M, N+1, N)
    return result if batched else result[0]
