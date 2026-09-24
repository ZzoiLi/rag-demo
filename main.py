import os
import shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from rag import VectorStore, load_document, split_text, build_prompt,split_by_heading

from rag import VectorStore, load_document, split_text

load_dotenv()

# ---------- LLM 客户端 ----------
llm_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ---------- 全局向量库 ----------
store = VectorStore()

UPLOAD_DIR = "data/uploads"

app = FastAPI(title="RAG Demo")


class ChatRequest(BaseModel):
    q: str


@app.get("/")
def root():
    return {"status": "ok", "chunks": store.count()}


# ---------- 文档上传 ----------
@app.post("/api/doc/upload")
async def upload_doc(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        text = load_document(save_path)
    except ValueError as e:
        return {"error": str(e)}

    if text.count("【") >= 5:
        chunks = split_by_heading(text)
    else:
        chunks = split_text(text)
    store.add(chunks)

    return {
        "filename": file.filename,
        "chars": len(text),
        "new_chunks": len(chunks),
        "total_chunks": store.count(),
    }


# ---------- 原问答接口（暂时还是直连 LLM） ----------
@app.post("/chat")
def chat(req: ChatRequest):
    hits = store.search(req.q, topk=6)

    if not hits:
        return {"answer": "知识库为空，请先上传文档。", "references": []}

    system, user = build_prompt(req.q, hits)

    resp = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )

    answer = resp.choices[0].message.content

    references = [
        {"id": i + 1, "score": round(score, 4), "text": chunk}
        for i, (chunk, score) in enumerate(hits)
    ]

    return {"answer": answer, "references": references}