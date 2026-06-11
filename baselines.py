import numpy as np
import torch
from typing import Union
from tqdm import tqdm

def l2_normalize_rows(x: Union[np.ndarray, torch.Tensor], eps: float = 1e-12) -> Union[np.ndarray, torch.Tensor]:
    """L2 normalize each row. Supports both numpy arrays and torch tensors."""
    if isinstance(x, torch.Tensor):
        x = x.float()
        norms = torch.norm(x, p=2, dim=1, keepdim=True)
        return x / (norms + eps)
    else:
        x = x.astype(np.float32, copy=False)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norms + eps)


def calculate_snr_curve(eigenvalues: np.ndarray, tail_start_index: int = None):
    """
    Compute Local SNR (marginal signal-to-noise ratio) at different truncation dimensions k.

    Local SNR formula: SNR_local(k) = (λ_k - σ²_noise) / σ²_noise = λ_k / σ²_noise - 1

    Args:
        eigenvalues: array of eigenvalues (sorted in descending order)
        tail_start_index: start index for estimating the noise floor, defaults to the last 10%

    Returns:
        snr_curve: SNR curve
        noise_variance: estimated noise variance
    """
    if tail_start_index is None:
        tail_start_index = int(len(eigenvalues) * 0.9)  # 90% of eigenvalues; tested 0.9, 0.95, 0.85, 0.8 — results are similar, within 0.0002 error

    # Estimate noise floor — assume the tail is pure noise
    noise_variance = np.mean(eigenvalues[tail_start_index:])

    # Compute Local SNR: marginal SNR for each dimension k
    snr_list = []
    for k in range(1, tail_start_index + 1):
        lambda_k = eigenvalues[k - 1]
        local_snr = (lambda_k - noise_variance) / noise_variance
        local_snr = max(0, local_snr)  # avoid negative values
        snr_list.append(local_snr)

    return np.array(snr_list), noise_variance


def optimal_gamma_by_kneedle(eigenvalues: np.ndarray, target_dim: int, S: float = 0.5) -> float:
    """
    Find the optimal gamma for spectral tempering by Kneedle algorithm.

    Approach:
    - Compute the SNR curve
    - Use Kneedle algorithm to find the knee point (signal-noise boundary)
    - best_gamma = SNR(target_dim) / SNR(knee_point)

    Args:
        eigenvalues: array of eigenvalues (sorted in descending order)
        target_dim: target dimension
        S: sensitivity parameter for Kneedle algorithm; smaller values are more sensitive

    Returns:
        best_gamma: optimal gamma value
    """
    from kneed import KneeLocator

    # 1. Compute SNR curve
    snr_curve, noise_variance = calculate_snr_curve(eigenvalues)

    # 2. Find knee point using Kneedle algorithm
    target_dims = np.arange(1, len(snr_curve) + 1)
    kneedle = KneeLocator(
        target_dims,
        snr_curve,
        curve='convex',           # SNR curve bends downward
        direction='decreasing',   # SNR decreases with dimension
        S=S                       # sensitivity parameter
    )

    knee_point = kneedle.knee
    if knee_point is None:
        print("[Warning] Kneedle could not find knee point, using default knee_point = target_dim // 2")
        knee_point = max(1, target_dim // 2)

    # 3. Retrieve SNR values
    # Ensure target_dim and knee_point are within valid range
    if target_dim > len(snr_curve):
        print(f"[Warning] target_dim ({target_dim}) > len(snr_curve) ({len(snr_curve)}), using last SNR value")
        snr_target = snr_curve[-1]
    else:
        snr_target = snr_curve[target_dim - 1]

    snr_knee = snr_curve[knee_point - 1]

    # 4. Compute best_gamma
    # Avoid division by zero
    if snr_knee == 0:
        print("[Warning] SNR(knee_point) is 0, returning gamma = 1.0")
        return 1.0

    best_gamma = snr_target / snr_knee

    print(f"[Kneedle] knee_point = {knee_point}, SNR(knee) = {snr_knee:.4f}")
    print(f"[Kneedle] target_dim = {target_dim}, SNR(target) = {snr_target:.4f}")
    print(f"[Kneedle] best_gamma = SNR({target_dim}) / SNR({knee_point}) = {best_gamma:.6f}")

    return best_gamma


class SpectralTemperingWhitener:

    def __init__(
        self,
        model: SentenceTransformer,
        data_folder: str,
        target_dim: int = 256,
        gamma: float = 0.05,
        auto_gamma: bool = True,
        kneedle_S: float = 0.5,
        epsilon: float = 1e-6,
        seed: int = 2026,
        data_loader_class=None,  # Pass GenericDataLoader here
    ):
        """Initializes the whitener by loading data, early text cropping to 1,000 documents,

        encoding them, and computing the static projection matrix via spectral analysis.
        """
        if data_loader_class is None:
            raise ValueError(
                "Please pass the 'GenericDataLoader' class reference to 'data_loader_class'."
            )

        print(f"Loading dataset from: {data_folder}...")
        corpus, queries, _ = data_loader_class(data_folder).load(split="test")

        # Extract raw texts
        doc_texts = [
            f"{corpus[did].get('title', '')} {corpus[did].get('text', '')}".strip()
            for did in corpus.keys()
        ]

        # Early text cropping to 1000 items to completely skip heavy encoding overhead
        np.random.seed(seed)
        if len(doc_texts) > 1000:
            crop_indices = np.random.choice(
                len(doc_texts), 1000, replace=False
            )
            doc_texts = [doc_texts[idx] for idx in crop_indices]

        print(f"Encoding {len(doc_texts)} calibration documents...")
        document_embeddings = model.encode(
            doc_texts,
            batch_size=256,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        d = document_embeddings.shape[1]

        # Cast to float64 for numerical precision
        sample_data = document_embeddings.astype(np.float64)
        self.sample_mean = np.mean(sample_data, axis=0)

        print("[*] Spectral Analysis on calibration sample...")
        cov_g = np.cov(sample_data.T)
        del sample_data

        L, U = np.linalg.eigh(cov_g)
        del cov_g

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(L)[::-1]
        L = L[idx]
        U = U[:, idx]

        # Auto-compute gamma (if enabled) using the kneedle function
        if auto_gamma:
            print(
                f"[*] Auto-calculating gamma using Kneedle algorithm (S={kneedle_S})..."
            )
            # Utilizing global scope function assumption
            self.final_gamma = optimal_gamma_by_kneedle(
                L, target_dim, S=kneedle_S
            )
            self.final_gamma = min(self.final_gamma, 1.0)
            print(f"    - Auto gamma calibrated to: {self.final_gamma:.6f}")
        else:
            self.final_gamma = gamma
            print(f"    - Using fixed gamma: {self.final_gamma}")

        # Compute scaling factors and bake them into the projection matrix
        scales = (L[:target_dim] + epsilon) ** (-self.final_gamma / 2.0)
        U_reduced = U[:, :target_dim]

        # Cached global transformation kernel
        self.P = U_reduced * scales[np.newaxis, :]
        print(f"[*] Physical Dimension Reduction Matrix Formed: {d} -> {target_dim}")

    def unified_spectral_tempering_truncation(
        self,
        query_embeddings: np.ndarray,
        document_embeddings: np.ndarray,
        target_dim: int = 256,  # Kept in signature for matching format
        gamma: float = 0.05,  # Kept in signature for matching format
        auto_gamma: bool = True,  # Kept in signature for matching format
        kneedle_S: float = 0.5,  # Kept in signature for matching format
        epsilon: float = 1e-6,  # Kept in signature for matching format
        seed: int = 2026,  # Kept in signature for matching format
        sample_size: int = 1000000,  # Kept in signature for matching format
        batch_size: int = 100000,
        remove_mean: bool = True,
    ):
        """Executes the batch spectral transform over target datasets using the projection matrix

        pre-calculated during instantiation.
        """
        n_docs = document_embeddings.shape[0]
        n_queries = query_embeddings.shape[0]

        # Batch transform documents
        transformed_docs_list = []
        for i in tqdm(range(0, n_docs, batch_size), desc="Spectral transform docs"):
            batch = document_embeddings[i : i + batch_size].astype(np.float64)

            if remove_mean:
                batch_transformed = np.dot(batch - self.sample_mean, self.P)
            else:
                batch_transformed = np.dot(batch, self.P)

            batch_transformed = l2_normalize_rows(batch_transformed)
            transformed_docs_list.append(batch_transformed.astype(np.float32))

        transformed_docs = np.vstack(transformed_docs_list)

        # Batch transform queries
        transformed_queries_list = []
        for i in tqdm(
            range(0, n_queries, batch_size), desc="Spectral transform queries"
        ):
            batch = query_embeddings[i : i + batch_size].astype(np.float64)

            if remove_mean:
                batch_transformed = np.dot(batch - self.sample_mean, self.P)
            else:
                batch_transformed = np.dot(batch, self.P)

            batch_transformed = l2_normalize_rows(batch_transformed)
            transformed_queries_list.append(batch_transformed.astype(np.float32))

        transformed_queries = np.vstack(transformed_queries_list)

        return transformed_queries, transformed_docs


class WhiteningKTruncationWhitener:

    def __init__(
        self,
        model: SentenceTransformer,
        data_folder: str,
        target_dim: int = 512,
        beta: float = 1.0,
        gamma: float = 1.0,
        seed: int = 2026,
        data_loader_class=None,  # Pass GenericDataLoader here
    ):
        """Initializes the whitener by loading data, early text cropping to 1,000 documents,

        encoding them, and computing the static target dim kernel projection.
        """
        if data_loader_class is None:
            raise ValueError(
                "Please pass the 'GenericDataLoader' class reference to 'data_loader_class'."
            )

        print(f"Loading dataset from: {data_folder}...")
        corpus, queries, _ = data_loader_class(data_folder).load(split="test")

        # Extract raw texts
        doc_texts = [
            f"{corpus[did].get('title', '')} {corpus[did].get('text', '')}".strip()
            for did in corpus.keys()
        ]

        # Early text cropping to 1000 items to completely bypass heavy encoding overhead
        np.random.seed(seed)
        if len(doc_texts) > 1000:
            crop_indices = np.random.choice(
                len(doc_texts), 1000, replace=False
            )
            doc_texts = [doc_texts[idx] for idx in crop_indices]

        print(f"Encoding {len(doc_texts)} calibration documents...")
        document_embeddings = model.encode(
            doc_texts,
            batch_size=256,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        n_dim = document_embeddings.shape[1]

        # 1. Process sample data (float64 for numerical precision)
        sample_data = document_embeddings.astype(np.float64)

        # Compute mean scaled by beta to control degree of mean subtraction
        self.sample_mean = np.mean(sample_data, axis=0) * beta

        # Compute covariance matrix (np.cov auto-centers)
        print("[*] Computing covariance and eigenvalue decomposition...")
        cov = np.cov(sample_data.T)
        del sample_data

        # 2. Eigenvalue decomposition
        L, U = np.linalg.eigh(cov)
        del cov

        # Sort in descending order
        idx = np.argsort(L)[::-1]
        L = L[idx]
        U = U[:, idx]

        # Take first target_dim principal components and build static whitening kernel
        L = L[:target_dim]
        U = U[:, :target_dim]
        eps = 1e-6
        self.kernel = U * ((L + eps) ** (-(gamma / 2.0)))
        print(f"[*] Whitening Kernel Matrix Formed: {n_dim} -> {target_dim}")

    def whitening_k_truncation(
        self,
        query_embeddings: np.ndarray,
        document_embeddings: np.ndarray,
        target_dim: int = 512,  # Kept in signature for matching format
        sample_size: int = 1000000,  # Kept in signature for matching format
        batch_size: int = 100000,
        seed: int = 2026,  # Kept in signature for matching format
        beta: float = 1.0,  # Unused inside class execution (pre-calculated)
        gamma: float = 1.0,  # Unused inside class execution (pre-calculated)
    ):
        """Executes the standard Whitening-k transform over downstream targets using the

        projection matrix and scaled mean pre-calculated during instantiation.
        """
        n_docs = document_embeddings.shape[0]
        n_queries = query_embeddings.shape[0]

        # 3. Batch transform documents
        transformed_docs_list = []
        for i in tqdm(range(0, n_docs, batch_size), desc="Whitening transform docs"):
            batch = document_embeddings[i : i + batch_size].astype(np.float64)
            batch_centered = batch - self.sample_mean
            batch_transformed = batch_centered @ self.kernel
            batch_transformed = l2_normalize_rows(batch_transformed)
            transformed_docs_list.append(batch_transformed.astype(np.float32))

        transformed_document_embeddings = np.vstack(transformed_docs_list)

        # 4. Batch transform queries
        transformed_queries_list = []
        for i in tqdm(
            range(0, n_queries, batch_size), desc="Whitening transform queries"
        ):
            batch = query_embeddings[i : i + batch_size].astype(np.float64)
            batch_centered = batch - self.sample_mean
            batch_transformed = batch_centered @ self.kernel
            batch_transformed = l2_normalize_rows(batch_transformed)
            transformed_queries_list.append(batch_transformed.astype(np.float32))

        transformed_query_embeddings = np.vstack(transformed_queries_list)

        return transformed_query_embeddings, transformed_document_embeddings
