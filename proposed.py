import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from beir.datasets.data_loader import GenericDataLoader


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """Helper to L2 normalize rows of a 2D array."""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + 1e-12)


class ZipfianSVDWhitener:

    def __init__(
        self,
        model: SentenceTransformer,
        data_folder: str,
        target_dim: int = 512,
        seed: int = 2026,
        data_loader_class=None,  # Pass GenericDataLoader here
    ):
        """Initializes the whitener by loading data, cropping text lists early to save compute,
        encoding the subset, and calibrating Zipfian decay parameters.
        """
        self.max_docs = 1000
        if data_loader_class is None:
            raise ValueError(
                "Please pass the 'GenericDataLoader' class reference to 'data_loader_class'."
            )

        print(f"Loading dataset from: {data_folder}...")
        corpus, queries, _ = data_loader_class(data_folder).load(split="test")

        # Extract textual fields
        doc_texts = [
            f"{corpus[did].get('title', '')} {corpus[did].get('text', '')}".strip()
            for did in corpus.keys()
        ]

        # Early structural crop to exactly 1000 items at the text level
        # to skip embedding overhead entirely.
        np.random.seed(seed)
        if len(doc_texts) > self.max_docs:
            crop_indices = np.random.choice(
                len(doc_texts), self.max_docs, replace=False
            )
            doc_texts = [doc_texts[idx] for idx in crop_indices]

        print(f"Encoding {len(doc_texts)} calibration documents...")
        document_embeddings = model.encode(
            doc_texts,
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        n_docs, n_dim = document_embeddings.shape
        sample_data = document_embeddings.astype(np.float64)

        # Split 1000 samples evenly into train and evaluation blocks
        opt_split = n_docs // 2
        train_sample = sample_data[:-opt_split]
        val_sample = sample_data[-opt_split:]

        # Empirical covariance and decomposition
        cov = np.cov(train_sample.T)

        L, U = np.linalg.eigh(cov)
        idx = np.argsort(L)[::-1]
        L = L[idx][:target_dim]
        U = U[:, idx][:, :target_dim]
        eps = 1e-6

        # 2. Unsupervised Optimization over Power-Law Decay Space
        best_score = -float("inf")
        self.best_beta = 1.0
        self.best_gamma = 1.0
        self.best_alpha = 0.2

        beta_grid = [0.5, 0.8, 1.0]
        gamma_grid = [0.6, 0.9, 1.2]
        alpha_grid = [0.1, 0.25, 0.5, 0.75]

        val_raw_mean = np.mean(val_sample, axis=0)
        indices = np.arange(1, target_dim + 1)

        # Uniformity evaluation matrix over validation slice
        v_sub_raw = l2_normalize_rows(val_sample)
        raw_sim = v_sub_raw @ v_sub_raw.T

        print("Optimizing hyper-parameters over validation slice...")
        for b in beta_grid:
            for g in gamma_grid:
                for a in alpha_grid:
                    gamma_vector = g * (1.0 / (indices**a))
                    scale_factors = (L + eps) ** (-(gamma_vector / 2.0))
                    candidate_kernel = U * scale_factors

                    v_centered = val_sample - (val_raw_mean * b)
                    v_trans = l2_normalize_rows(v_centered @ candidate_kernel)

                    sim_matrix = v_trans @ v_trans.T
                    uniformity_loss = np.log(np.mean(np.exp(sim_matrix * 2.0)))
                    alignment_loss = np.mean((sim_matrix - raw_sim) ** 2)

                    score = -uniformity_loss - (3.5 * alignment_loss)

                    if score > best_score:
                        best_score = score
                        self.best_beta = b
                        self.best_gamma = g
                        self.best_alpha = a

        print(
            f"Calibration Complete! Best Beta: {self.best_beta}, Gamma: {self.best_gamma}, Alpha: {self.best_alpha}"
        )

        # 3. Assemble and Cache Global Transformation Matrix
        self.final_mean = np.mean(sample_data, axis=0) * float(self.best_beta)
        final_cov = np.cov(sample_data.T)

        L_f, U_f = np.linalg.eigh(final_cov)
        idx_f = np.argsort(L_f)[::-1]
        L_f = L_f[idx_f][:target_dim]
        U_f = U_f[:, idx_f][:, :target_dim]

        final_gamma_vector = float(self.best_gamma) * (
            1.0 / (indices ** float(self.best_alpha))
        )
        final_scale_factors = (L_f + eps) ** (-(final_gamma_vector / 2.0))

        # Cached global kernel transformation matrix
        self.kernel = U_f * final_scale_factors

    def automatic_optimized_zipfian_svd(
        self,
        query_embeddings: np.ndarray,
        document_embeddings: np.ndarray,
        target_dim: int = 512,
        sample_size: int = 1000000,
        batch_size: int = 100000,
        seed: int = 2026,
        beta: float = 1.0,  # Unused inside class execution (pre-calculated)
        gamma: float = 1.0,  # Unused inside class execution (pre-calculated)
    ):
        """Executes the Power-Law Regularized Whitening-k Transform using the parameters
        calibrated during class initialization. Signature matches original requirements.
        """
        n_docs = document_embeddings.shape[0]
        n_queries = query_embeddings.shape[0]

        # 4. Batch Execution Matrix Multiplication (Documents)
        transformed_docs_list = []
        for i in tqdm(range(0, n_docs, batch_size), desc="Power-Law Whitening docs"):
            batch = document_embeddings[i : i + batch_size].astype(np.float64)
            batch_centered = batch - self.final_mean
            batch_transformed = batch_centered @ self.kernel
            batch_transformed = l2_normalize_rows(batch_transformed)
            transformed_docs_list.append(batch_transformed.astype(np.float32))

        transformed_document_embeddings = np.vstack(transformed_docs_list)

        # 5. Batch Execution Matrix Multiplication (Queries)
        transformed_queries_list = []
        for i in tqdm(
            range(0, n_queries, batch_size), desc="Power-Law Whitening queries"
        ):
            batch = query_embeddings[i : i + batch_size].astype(np.float64)
            batch_centered = batch - self.final_mean
            batch_transformed = batch_centered @ self.kernel
            batch_transformed = l2_normalize_rows(batch_transformed)
            transformed_queries_list.append(batch_transformed.astype(np.float32))

        transformed_query_embeddings = np.vstack(transformed_queries_list)

        return transformed_query_embeddings, transformed_document_embeddings
