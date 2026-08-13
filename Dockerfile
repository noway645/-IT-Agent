# 企业 IT 助手 - 生产镜像
# 多阶段构建：builder 装依赖（利用层缓存），runtime 仅复制产物
FROM python:3.11-slim AS builder

# 编译依赖（psycopg2 需 libpq-dev + gcc；bcrypt 需 gcc）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
# 装到独立目录，便于 runtime 阶段整体复制
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim AS runtime

# 运行时仅需 libpq5（psycopg2 动态链接）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -r -u 1000 -m app

# 复制依赖
COPY --from=builder /install /usr/local

WORKDIR /app

# 复制源码
COPY --chown=app:app app.py db.py tools.py rag.py manage.py migrate_sqlite_to_pg.py ./
COPY --chown=app:app knowledge_base ./knowledge_base

USER app

EXPOSE 7860

# 健康检查：Gradio 启动后 /health 返回 200
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:7860/health',timeout=3).status==200 else sys.exit(1)"

CMD ["python", "app.py"]
