from collections import defaultdict
import numpy as np
import pandas as pd
import traceback

retriever = EvaluateRetrieval()


import pandas as pd
import numpy as np
import traceback

def evaluate_dimension_reduction(
    methods_to_test,
    q_emb,
    d_emb,
    scores_full,
    query_ids,
    doc_ids,
    qrels,
    retriever,
    prop_whitener,
    target_dim=192
):
    """
    Evaluates vector dimension reduction and quantization methods against a full baseline.
    Uses the modern composition configuration with a Modular UniformQuantizer backend.
    """

    # -------------------------------------------------------------------------
    # INNER HELPER: Scores to Results (Fast)
    # -------------------------------------------------------------------------
    def _scores_to_results_fast(scores, q_ids, d_ids, max_k=1000):
        if hasattr(scores, "detach"):
            scores = scores.detach().cpu().numpy()

        q_strs = [str(q) for q in q_ids]
        d_strs = [str(d) for d in d_ids]
        d_strs_arr = np.array(d_strs)

        k = min(max_k, scores.shape[1])
        top_k_idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]

        row_indices = np.arange(scores.shape[0])[:, None]
        top_k_scores = scores[row_indices, top_k_idx]
        sorted_top_k_meta_idx = np.argsort(-top_k_scores, axis=1)

        final_top_k_idx = top_k_idx[row_indices, sorted_top_k_meta_idx]

        results = {}
        for i, q_str in enumerate(q_strs):
            idx_list = final_top_k_idx[i]
            results[q_str] = dict(zip(d_strs_arr[idx_list], scores[i, idx_list].tolist()))

        return results

    # -------------------------------------------------------------------------
    # 1. EVALUATE BASELINE (FULL)
    # -------------------------------------------------------------------------
    results_full = _scores_to_results_fast(scores_full, query_ids, doc_ids)
    ndcg_full, map_full, _, _ = retriever.evaluate(qrels, results_full, retriever.k_values)

    rows = []
    for k in retriever.k_values:
        rows.append({"Metric": f"NDCG@{k}", "Full": ndcg_full[f"NDCG@{k}"]})
        rows.append({"Metric": f"MAP@{k}", "Full": map_full[f"MAP@{k}"]})

    df_results = pd.DataFrame(rows).set_index("Metric")

    # -------------------------------------------------------------------------
    # 2. EVALUATE EXPERIMENTAL METHODS
    # -------------------------------------------------------------------------
    for name, method_func in methods_to_test.items():
        try:
            q_proj, d_proj = method_func(q_emb, d_emb)

            # --- Float Case (Unquantized) ---
            if d_proj.dtype != np.uint8:
                qn = q_proj / (np.linalg.norm(q_proj, axis=1, keepdims=True) + 1e-8)
                dn = d_proj / (np.linalg.norm(d_proj, axis=1, keepdims=True) + 1e-8)
                scores_proj = qn @ dn.T

            # --- Quantized Array / Bitstream Demultiplexing via Modular Object ---
            else:
                n_q, n_d = q_proj.shape[0], d_proj.shape[0]
                bytes_per_row_d = d_proj.shape[1]
                bits = int(round((bytes_per_row_d * 8) / target_dim))

                # Utilize the class instance quantizer logic
                quantizer = getattr(prop_whitener, "quantizer", UniformQuantizer())

                # Unpack Documents
                d_recon = quantizer.unpack(d_proj, n_d * target_dim, target_dim, bits)
                if hasattr(prop_whitener, "doc_alphas"):
                    scale = prop_whitener.doc_alphas[:n_d]
                    d_recon *= scale

                # Unpack Queries
                if q_proj.dtype == np.uint8:
                    qn = quantizer.unpack(q_proj, n_q * target_dim, target_dim, bits)
                else:
                    qn = q_proj / (np.linalg.norm(q_proj, axis=1, keepdims=True) + 1e-8)

                scores_proj = qn @ d_recon.T

            # --- Eval & Populate Dataframe ---
            results_proj = _scores_to_results_fast(scores_proj, query_ids, doc_ids)
            ndcg_proj, map_proj, _, _ = retriever.evaluate(qrels, results_proj, retriever.k_values)

            for k in retriever.k_values:
                df_results.loc[f"NDCG@{k}", name] = ndcg_proj[f"NDCG@{k}"]
                df_results.loc[f"MAP@{k}", name] = map_proj[f"MAP@{k}"]

        except Exception as e:
            print(f"Error evaluating method '{name}': {e}")
            traceback.print_exc()
            df_results[name] = np.nan

        # Calculate retention percentage safely
        full_vals = df_results["Full"]
        proj_vals = df_results[name] if name in df_results.columns else np.nan
        df_results[f"{name} Ret (%)"] = np.where(full_vals > 0, (proj_vals / full_vals) * 100, 0.0)

    # -------------------------------------------------------------------------
    # 3. CALCULATE AVERAGES
    # -------------------------------------------------------------------------
    # Computes the mean for all value columns and retention percentage columns
    avg_row = df_results.mean(axis=0)
    df_results.loc["Average"] = avg_row

    # -------------------------------------------------------------------------
    # 4. FORMAT FINAL OUTPUT TABLE
    # -------------------------------------------------------------------------
    df_results = df_results.reset_index()

    column_order = ["Metric", "Full"]
    for name in methods_to_test.keys():
        column_order.extend([name, f"{name} Ret (%)"])

    df_results = df_results[[c for c in column_order if c in df_results.columns]]
    return df_results
