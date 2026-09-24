import json
import numpy as np
from rag import VectorStore

store = VectorStore()
print(f"总 chunks: {len(store.chunks)}")
print(f"BM25 索引: {'OK' if store.bm25 else 'None'}")
print()

query = "达英-35的适应症是什么"
print(f"Query: {query}")
print()

# 原始向量分，用于对比
import numpy as np
from rag import embed

q = embed([query])[0]
q = q / np.linalg.norm(q)
vec_scores = store.vectors @ q

target_idx = [i for i, c in enumerate(store.chunks) if "适应症" in c]
print(f"含'适应症'的 chunk 索引: {target_idx}")
for i in target_idx:
    print(f"  chunk {i}: 纯向量 sim = {vec_scores[i]:.4f}")
print()

hits = store.search(query, topk=6)
print("混合检索 Top 6:")
for rank, (chunk, score) in enumerate(hits, 1):
    marker = " ⭐" if "适应症" in chunk else ""
    print(f"  {rank}. vec_sim={score:.4f}{marker}")
    print(f"     {chunk[:70].replace(chr(10), ' ')}...")