
import os
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

KNOWLEDGE_BASE_PATH = os.path.join(
    BASE_DIR,
    "knowledge_base.jsonl"
)


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

with open(
    KNOWLEDGE_BASE_PATH,
    "r",
    encoding="utf-8"
) as f:

    records = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


# ============================================================
# PREPARE DOCUMENTS
# ============================================================

documents = []

for i, record in enumerate(records):

    text = f"""
Crop: {record['crop']}
Disease: {record['disease']}
Topic: {record['topic']}

Information:
{record['content']}

Source:
{record['source_title']}

Organization/Author:
{record['source_organization']}

Publication year:
{record['publication_year']}

Source type:
{record['source_type']}

Region:
{record['region']}

Evidence type:
{record['evidence_type']}

Confidence:
{record['confidence']}
"""

    documents.append({
        "id": i,
        "text": text.strip(),
        "metadata": record
    })


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

document_texts = [
    document["text"]
    for document in documents
]

embeddings = embedding_model.encode(
    document_texts,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")


# ============================================================
# CREATE FAISS INDEX
# ============================================================

embedding_dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    embedding_dimension
)

index.add(
    embeddings
)


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve(query, top_k=5):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    top_k = min(
        top_k,
        index.ntotal
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        results.append({
            "score": float(score),
            "document": documents[idx]
        })

    return results


# ============================================================
# STATUS
# ============================================================

print(
    f"Loaded {len(records)} knowledge records."
)

print(
    f"Created FAISS index with {index.ntotal} vectors."
)
