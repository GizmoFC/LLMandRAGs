"""
Embedding Generation Script
==========================

This script loads JSONL files containing contract clauses,
generates embeddings using sentence-transformers,
and stores them in a FAISS index for fast similarity search.

Usage:
    python embed_clauses.py

Dependencies:
    - sentence-transformers
    - faiss-cpu
    - jsonlines
    - numpy
"""

import json
import os
from tqdm import tqdm
from pathlib import Path
import faiss
import numpy as np

# Choose an embedding model
from sentence_transformers import SentenceTransformer

# ---- Step 1: Load clauses from JSONL ----

def load_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

# ---- Step 2: Build and embed texts ----

def build_embedding_text(entry):
    # Combine clause type, clause text, and context
    return f"{entry['clause_type']}\n{entry['text']}\n\n{entry['full_context']}"

# ---- Step 3: Embed and store in FAISS ----

def embed_clauses(jsonl_path, output_dir):
    print(f"Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast and light; good for semantic search

    print(f"Reading clauses...")
    entries = list(load_jsonl(jsonl_path))
    texts = [build_embedding_text(e) for e in entries]
    metadata = [{"clause_type": e["clause_type"], "source": e["source"], "title": e["contract_title"]} for e in entries]

    print(f"Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    print(f"Creating FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))

    print(f"Saving index and metadata...")
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "cuad_faiss.index"))

    with open(os.path.join(output_dir, "cuad_metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"Done! {len(texts)} entries embedded.")

# ---- Main ----

if __name__ == "__main__":
    jsonl_path = "cuad_clause_library.jsonl"
    output_dir = "embeddings/output"
    embed_clauses(jsonl_path, output_dir)


