import os
import json
import numpy as np
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import jieba
from rank_bm25 import BM25Okapi

load_dotenv()

# ============ Embedding（硅基流动） ============
_embed_client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)

EMBED_MODEL = "BAAI/bge-m3"


def embed(texts: list[str]) -> np.ndarray:
    """把一批文本转成向量，返回 shape=(n, 1024) 的数组"""
    resp = _embed_client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.array(vecs, dtype=np.float32)


# ============ 文档解析 ============
def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

import re

def clean_text(text: str) -> str:
    """清理 PDF 提取的脏文本"""
    # 1. 中文/英文之间被换行拆散的，合并回去
    #    "中\n方\n斡旋" -> "中方斡旋"
    text = re.sub(r'(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])', '', text)
    #    "fric\ntion" -> "friction"
    text = re.sub(r'(?<=[A-Za-z])\n(?=[A-Za-z])', '', text)

    # 2. 去掉形如 0.016*"friction" 的图表权重噪声
    text = re.sub(r'\d+\.\d+\*"[^"]*"', '', text)

    # 3. 多个连续换行合成一个
    text = re.sub(r'\n{2,}', '\n', text)

    # 4. 行首行尾多余空格
    text = "\n".join(line.strip() for line in text.split("\n"))

    return text

def load_document(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".pdf"):
        return clean_text(load_pdf(path))
    if lower.endswith(".txt"):
        return clean_text(load_txt(path))
    raise ValueError(f"不支持的文件类型: {path}")


# ============ 文本切分 ============
def split_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """按字符数滑动窗口切分，带 overlap"""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks

def split_by_heading(text: str) -> list[str]:
    """按【xxx】标题切分，适合说明书 / 手册类文档"""
    parts = re.split(r'(?=【[^】]{1,10}】)', text)
    chunks = []
    for p in parts:
        p = p.strip()
        if len(p) < 20:
            continue
        # 超长的小节再滑窗切
        if len(p) > 800:
            chunks.extend(split_text(p, chunk_size=500, overlap=80))
        else:
            chunks.append(p)
    return chunks


# ============ 向量存储 + 检索 ============
class VectorStore:
    def __init__(self, dir_path: str = "data/store"):
        self.dir = dir_path
        self.chunks_path = os.path.join(dir_path, "chunks.json")
        self.vectors_path = os.path.join(dir_path, "vectors.npy")
        self.chunks: list[str] = []
        self.vectors: np.ndarray | None = None
        self.bm25: BM25Okapi | None = None
        self._load()
        self._build_bm25()

    def _load(self):
        if os.path.exists(self.chunks_path) and os.path.exists(self.vectors_path):
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            self.vectors = np.load(self.vectors_path)

    def _build_bm25(self):
        """重建 BM25 索引"""
        if not self.chunks:
            self.bm25 = None
            return
        tokenized = [list(jieba.cut(c)) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def _save(self):
        os.makedirs(self.dir, exist_ok=True)
        with open(self.chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        np.save(self.vectors_path, self.vectors)

    def add(self, chunks: list[str]):
        if not chunks:
            return
        vecs = embed(chunks)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        if self.vectors is None:
            self.vectors = vecs
        else:
            self.vectors = np.vstack([self.vectors, vecs])
        self.chunks.extend(chunks)
        self._save()
        self._build_bm25()          # 新增后重建 BM25

    def search(self, query: str, topk: int = 4) -> list[tuple[str, float]]:
        """BM25 + 向量混合检索"""
        if self.vectors is None or not self.chunks or self.bm25 is None:
            return []

        # ---- 1. 向量分数 ----
        q = embed([query])[0]
        q = q / np.linalg.norm(q)
        vec_scores = self.vectors @ q

        # ---- 2. BM25 分数 ----
        tokenized_query = list(jieba.cut(query))
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query), dtype=np.float32)

        # ---- 3. 各自 min-max 归一化到 [0,1] ----
        def norm(x):
            lo, hi = x.min(), x.max()
            if hi - lo < 1e-8:
                return np.zeros_like(x)
            return (x - lo) / (hi - lo)

        vec_n = norm(vec_scores)
        bm25_n = norm(bm25_scores)

        # ---- 4. 加权融合：向量 0.5 + BM25 0.5 ----
        final = 0.2 * vec_n + 0.8 * bm25_n
        idx = np.argsort(-final)[:topk]

        # 返回原始向量分用于展示（分数含义统一）
        return [(self.chunks[i], float(final[i])) for i in idx]

    def count(self) -> int:
        return len(self.chunks)

def build_prompt(question: str, hits: list[tuple[str, float]]) -> tuple[str, str]:
    """拼 RAG prompt，返回 (system, user)"""
    context = "\n\n".join(
        f"[{i + 1}] {chunk}" for i, (chunk, _) in enumerate(hits)
    )
    system = (
        "你是一个严谨的文档问答助手。请严格依据【资料】回答用户问题。\n"
        "规则：\n"
        "1. 只使用资料中的信息，不要编造资料之外的内容。\n"
        "2. 如果资料中没有相关信息，直接回答“根据已上传的文档，我找不到相关信息”。\n"
        "3. 回答时在句末用 [1]、[2] 这样的编号标注你参考了哪段资料。\n"
        "4. 用中文回答，简洁清晰。"
    )
    user = f"【资料】\n{context}\n\n【问题】\n{question}"
    return system, user