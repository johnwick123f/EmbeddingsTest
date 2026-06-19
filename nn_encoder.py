import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from scipy.cluster.vq import kmeans2
from quantizer import UniformQuantizer

# --- Official ConvNeXt V2 Component Integrations ---

class LayerNorm(nn.Module):
    """ Official ConvNeXt V2 LayerNorm supporting channels_last and channels_first natively. """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ Official ConvNeXt V2 GRN (Global Response Normalization) layer.
    Expects channels_last format: (B, H, W, C)
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class DropPath(nn.Module):
    """ Clean, explicit implementation of Stochastic Depth / Drop Path. """
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x / keep_prob * binary_tensor

class GRN1d(nn.Module):
    """
    Global Response Normalization optimized for 1D Vector Tensors (Batch, Channels).
    Forces channels to compete, drastically sharpening feature contrast.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2 norm across the batch/spatial dimension to compute global channel response
        # x shape: (B, C) -> norm shape: (1, C)
        gx = torch.norm(x, p=2, dim=0, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return x * (1 + self.gamma * nx) + self.beta

class ProjectionNet(nn.Module):
    def __init__(self, in_features: int, out_features: int, drop_path: float = 0.01):
        """
        Premium Autoencoder Bottleneck Projector tailored for RabbitQ Quantization.
        Bounds scaling envelopes and stabilizes cross-feature competitive variance.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        hidden_dim = max(in_features, out_features * 2)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # --- 1D Path Setup ---
        self.norm1d_1 = nn.LayerNorm(in_features, eps=1e-6)
        self.proj_in1d = nn.Linear(in_features, hidden_dim)
        self.proj_gate1d = nn.Linear(in_features, hidden_dim)

        self.grn1d = GRN1d(hidden_dim)
        self.norm1d_2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.proj_out1d = nn.Linear(hidden_dim, out_features)

        self.shortcut1d = (
            nn.Linear(in_features, out_features, bias=False)
            if in_features != out_features else nn.Identity()
        )

        # --- Soft Bounds Bounding Activations for Uniform Bins ---
        # Replaced GELU with Tanh on the gating path to tightly control scale explosions
        self.gate_act = nn.Tanh()

    def _forward_2d_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vector Processing Pipeline.
        Features bounded channel competition to prevent uniform bin clipping.
        """
        norm_x = self.norm1d_1(x)

        # Bounded GLU protects uniform quantization boundaries
        gate = self.gate_act(self.proj_gate1d(norm_x))
        features = self.proj_in1d(norm_x) * gate

        # Inject feature competition
        features = self.grn1d(features)
        features_norm = self.norm1d_2(features)

        out = self.proj_out1d(features_norm)

        # Latent Space residual mix
        latent = self.shortcut1d(x) + self.drop_path(out)

        # --- THE L2 HYPERSPHERE BRIDGE ---
        # Ensures output matches RabbitQ's implicit Unit-Norm design perfectly
        return torch.nn.functional.normalize(latent, p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self._forward_2d_tensor(x)
        else:
            raise ValueError(f"Expected 1D feature vectors (Batch, Features) with dim==2, got {x.dim()}")

def spherical_uniformity_loss(x_norm, t=8):
    """
    Pushes normalized vectors to distribute uniformly across the unit hypersphere surface.
    Returns a positive loss value where lower is better (0 = perfectly spread out to infinity).
    """
    sq_distances = torch.cdist(x_norm, x_norm, p=2) ** 2

    # Just calculate the raw RBF kernel score.
    # If points are clustered, this climbs toward 1.0. If spread out, drops toward 0.
    return torch.exp(-t * sq_distances).mean()

# -------------------------------------------------------------------------
# Self-contained Hybrid Muon Optimizer (No external libraries required)
# -------------------------------------------------------------------------
class HybridMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.05, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)
        
        # Internal AdamW tracking for 1D parameters (biases/gains)
        self.adamw_state = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure != None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            momentum = group['momentum']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # --- 1D Parameters (Biases, Weights with dim < 2) -> Optimized via AdamW ---
                if p.ndim < 2:
                    if 'step' not in state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    
                    state['step'] += 1
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    
                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(0.9).add_(grad, alpha=0.1)
                    exp_avg_sq.mul_(0.999).addcmul_(grad, grad, value=0.001)
                    
                    bias_correction1 = 1 - 0.9 ** state['step']
                    bias_correction2 = 1 - 0.999 ** state['step']
                    
                    # Apply weight decay
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                        
                    denom = (exp_avg_sq.sqrt() / np.sqrt(bias_correction2)).add_(1e-8)
                    step_size = lr / bias_correction1
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    continue

                # --- 2D Parameters (Matrix Weights) -> Optimized via Muon ---
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)

                # Newton-Schulz Orthogonalization of the update
                g = buf
                X = g / (g.norm() + 1e-7)
                
                # Check if we need to transpose to ensure wide matrix form for Newton-Schulz
                transposed = False
                if X.shape[0] > X.shape[1]:
                    X = X.t()
                    transposed = True

                for _ in range(ns_steps):
                    A = torch.mm(X, X.t())
                    B = A @ X
                    X = 3.4445 * X - 4.775 * B + 2.0315 * A @ B

                if transposed:
                    X = X.t()

                # Apply weight decay and Muon update step
                if wd != 0:
                    p.mul_(1 - lr * wd)
                
                # Muon updates use a scaled internal learning rate relative to output dimensions
                scale = max(1, p.shape[0]) ** 0.5
                p.add_(X, alpha=-lr * scale)

        return loss

class QuantizedWhitener:
    def __init__(
        self,
        model,
        data_folder: str,
        target_dim: int = 512,
        beta: float = 1.0,
        gamma: float = 1.0,
        seed: int = 2026,
        data_loader_class=None,
    ):
        if data_loader_class is None:
            raise ValueError("Please pass 'GenericDataLoader' to 'data_loader_class'.")

        print(f"Loading dataset from: {data_folder}...")
        corpus, _, _ = data_loader_class(data_folder).load(split="test")

        doc_texts = [
            f"{corpus[did].get('title', '')} {corpus[did].get('text', '')}".strip()
            for did in corpus.keys()
        ]

        np.random.seed(seed)
        torch.manual_seed(seed)

        if len(doc_texts) > 40000:
            doc_texts = [doc_texts[i] for i in np.random.choice(len(doc_texts), 20000, replace=False)]

        print(f"Encoding {len(doc_texts)} calibration documents...")
        document_embeddings = model.encode(
            doc_texts,
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        n_dim = document_embeddings.shape[1]
        sample_data = document_embeddings.astype(np.float32)

        self.sample_mean = np.mean(sample_data, axis=0)
        centered_data = sample_data - self.sample_mean

        # -------------------------
        # Projection model
        # -------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = target_dim
        self.half = target_dim // 2

        self.quantizer = UniformQuantizer()
        self.projection_model = ProjectionNet(n_dim, target_dim).to(self.device)

        # REPLACED: Using our native HybridMuon optimizer
        optimizer = HybridMuon(self.projection_model.parameters(), lr=1e-3, weight_decay=0.05)

        # T_max is set to total epochs (100) so it reaches eta_min at the final epoch
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=9e-4)

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(centered_data)),
            batch_size=1024,
            shuffle=True,
            drop_last=True,
        )

        print("[*] Training network via Mean-Centered Cosine Distance Alignment...")
        self.projection_model.train()

        for epoch in range(20):
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)

                optimizer.zero_grad()

                bx_n = batch_x / (batch_x.norm(dim=1, keepdim=True) + 1e-8)
                target_sim = torch.mm(bx_n, bx_n.t())

                encoded = self.projection_model(batch_x)

                enc_n = encoded / (encoded.norm(dim=1, keepdim=True) + 1e-8)
                pred_sim = torch.mm(enc_n, enc_n.t())
                reg_loss = spherical_uniformity_loss(enc_n)

                loss = nn.functional.mse_loss(pred_sim - pred_sim.mean(), target_sim - target_sim.mean()) + (reg_loss * 0.05)
                loss.backward()
                optimizer.step()

            scheduler.step()

            if epoch % 25 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}, Loss: {loss:.6f}, LR: {current_lr:.2e}")

        self.projection_model.eval()
        torch.cuda.empty_cache()

    def _validate_quantization_params(self, target_dim: int, bits: int):
        if bits not in [1, 2, 3, 4, 5, 8]:
            raise ValueError("Supported bit widths are 1, 2, 3, 4, or 5 to remain byte-aligned.")

        if (target_dim * bits) % 8 != 0:
            raise ValueError(
                f"target_dim ({target_dim}) with {bits}-bit quantization "
                f"results in a fractional byte row length ({ (target_dim * bits) / 8 }). "
                f"Please ensure (target_dim * bits) is a multiple of 8."
            )

    def compress_embeddings(
        self,
        query_embeddings: np.ndarray,
        document_embeddings: np.ndarray,
        target_dim: int = 512,
        batch_size: int = 16384,
        seed: int = 2026,
        beta: float = 1.0,
        gamma: float = 1.0,
        bits: int = 2,
        quantize: bool = True,
    ):
        if quantize:
            self._validate_quantization_params(target_dim, bits)

        # =========================
        # QUERIES
        # =========================
        tx_queries = []
        for i in range(0, query_embeddings.shape[0], batch_size):
            batch = query_embeddings[i:i+batch_size].astype(np.float32) - self.sample_mean
            with torch.no_grad():
                proj = self.projection_model(torch.from_numpy(batch).to(self.device)).cpu().numpy()
            proj = l2_normalize_rows(proj)

            if quantize:
                # Outsource logic directly to the external UniformQuantizer setup
                tx_queries.append(self.quantizer.quantize_and_pack(proj, bits=bits))
            else:
                tx_queries.append(proj)
        tx_queries = np.vstack(tx_queries)

        # =========================
        # DOCS
        # =========================
        tx_docs = []
        for i in range(0, document_embeddings.shape[0], batch_size):
            batch = document_embeddings[i:i+batch_size].astype(np.float32) - self.sample_mean
            with torch.no_grad():
                proj = self.projection_model(torch.from_numpy(batch).to(self.device)).cpu().numpy()
            proj = l2_normalize_rows(proj)

            if quantize:
                # Outsource logic directly to the external UniformQuantizer setup
                tx_docs.append(self.quantizer.quantize_and_pack(proj, bits=bits))
            else:
                tx_docs.append(proj)
        tx_docs = np.vstack(tx_docs)

        return tx_queries, tx_docs
