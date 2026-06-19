import random
import numpy as np
import pandas as pd
import gc
import torch

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

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
import gc
import torch
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
