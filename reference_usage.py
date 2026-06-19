import random
import numpy as np
import pandas as pd
import gc
import torch

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from nn_encoder import QuantizedWhitener
from spec_svd import QuantizedWhitenerSpec 
from eval import evaluate_dimension_reduction

def l2_normalize_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)
    
## Get this from here https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip and unzip

model = SentenceTransformer(
    "mixedbread-ai/mxbai-embed-large-v1",
    model_kwargs={"torch_dtype": "float16"},
)
corpus, queries, qrels = GenericDataLoader(
    data_folder="/content/EmbeddingsTest/scifact"
).load(split="test")

query_ids = list(queries.keys())
query_texts = [queries[qid] for qid in query_ids]

doc_ids = list(corpus.keys())
doc_texts = [
    f"{corpus[did].get('title', '')} {corpus[did].get('text', '')}".strip()
    for did in doc_ids
]

print(f"{len(query_texts):,} queries")
print(f"{len(doc_texts):,} documents")

gc.collect()
torch.cuda.empty_cache()

print("Encoding queries...")
q_emb = model.encode(
    query_texts,
    batch_size=128,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)

print("Encoding documents...")
d_emb = model.encode(
    doc_texts,
    batch_size=128,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)

print("Computing baseline scores...")
scores_full = q_emb @ d_emb.T

gc.collect()
torch.cuda.empty_cache()

target_dim = 256

prop_encoder = QuantizedWhitener(
    model=model,
    data_folder="/content/EmbeddingsTest/quora",
    target_dim=target_dim,
    data_loader_class=GenericDataLoader,
)
spec_encoder = QuantizedWhitenerSpec(
    model=model,
    data_folder="/content/EmbeddingsTest/quora",
    target_dim=target_dim,
    data_loader_class=GenericDataLoader,

retriever = EvaluateRetrieval()


# 1. Store the specific whitener object right inside the config dict
methods_to_test = [
    {
        "name": "Proposed Autoencoder",
        "whitener": prop_encoder,
        "func": lambda q, d: prop_encoder.compress_embeddings(q, d, target_dim=target_dim, bits=2)
    },
    {
        "name": "Spectral SVD",
        "whitener": spec_encoder,
        "func": lambda q, d: spec_encoder.compress_embeddings(q, d, target_dim=target_dim, bits=2)
    }
]

# 2. Clean, dynamic loop without a single if/else statement
for method in methods_to_test:
    df_comparison = evaluate_dimension_reduction(
        methods_to_test={method["name"]: method["func"]}, # Keeps your original function format happy
        q_emb=q_emb,
        d_emb=d_emb,
        scores_full=scores_full,
        query_ids=query_ids,
        doc_ids=doc_ids,
        qrels=qrels,
        retriever=retriever,
        prop_whitener=method["whitener"],                  # Dynamic lookup!
        target_dim=target_dim
    )

    print(df_comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notnull(x) else "NaN"))
