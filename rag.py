"""RAG 知识库模块 - 阶段二

负责：加载 Markdown 文档 → 切块 → 智谱 embedding-3 向量化 → Chroma 持久化 → 检索。
向量库本地存于 ./chroma_db（已 gitignore），首次构建后直接复用。
"""
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载 .env 中的 ZHIPUAI_API_KEY
load_dotenv()

# 路径
BASE_DIR = Path(__file__).parent
KB_DIR = BASE_DIR / "knowledge_base"
DB_DIR = BASE_DIR / "chroma_db"

# 参数
EMBED_MODEL = "embedding-3"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
SEARCH_K = 4
# 智谱 embedding 单次 input 上限 64 条，需分批入库
EMBED_BATCH = 64

# 检索增强回答的系统提示词
SYSTEM_PROMPT = (
    "你是星辰科技 IT 部门的智能助手，负责帮助员工解决日常 IT 问题。"
    "请仅基于下方 <context> 中检索到的知识库文档回答员工问题，"
    "回答用中文，分步骤、清晰、可操作。\n"
    "若文档中没有相关信息，请明确告知'知识库中暂无此问题答案'，"
    "并建议联系 IT 服务台（电话 8000，邮箱 it@company.com）。"
    "禁止编造文档中未出现的内容或参数。\n\n"
    "<context>\n{context}\n</context>"
)


def _load_docs():
    """加载 knowledge_base 下全部 Markdown 文档。"""
    docs = []
    for md in sorted(KB_DIR.glob("*.md")):
        docs.extend(TextLoader(str(md), encoding="utf-8").load())
    return docs


def _get_embeddings():
    return ZhipuAIEmbeddings(model=EMBED_MODEL)


def build_vectorstore():
    """首次构建：加载 → 切块 → 分批入库 → 持久化。"""
    docs = _load_docs()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    embeddings = _get_embeddings()
    # 智谱 embedding 单次最多 64 条，分批写入
    vs = None
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        if vs is None:
            vs = Chroma.from_documents(
                batch, embeddings, persist_directory=str(DB_DIR)
            )
        else:
            vs.add_documents(batch)
    return vs, len(chunks)


def get_retriever():
    """优先复用已持久化的向量库；不存在则构建。"""
    if DB_DIR.exists() and any(DB_DIR.iterdir()):
        vs = Chroma(
            persist_directory=str(DB_DIR), embedding_function=_get_embeddings()
        )
    else:
        vs, _ = build_vectorstore()
    return vs.as_retriever(search_kwargs={"k": SEARCH_K})
