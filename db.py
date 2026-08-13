"""数据库模块 - 账号/工单/审计/对话（PostgreSQL 版）

P0 上线改造：SQLite → PostgreSQL + 连接池。
- 保持所有函数签名不变，app.py/tools.py/manage.py 无需改动
- 时间戳仍用 TEXT 存 ISO8601 字符串（PG 中 lexicographic 排序正确）
- 连接池：psycopg2.pool.SimpleConnectionPool，minconn=1 maxconn=10
- schema 新增常用查询索引（username/ts）

环境变量：
  PG_HOST (默认 localhost)
  PG_PORT (默认 5432)
  PG_DB   (默认 it_helper)
  PG_USER (默认 it_helper)
  PG_PASSWORD
  PG_POOL_MIN (默认 1)
  PG_POOL_MAX (默认 10)
"""
import os
from contextlib import contextmanager
from datetime import datetime

import bcrypt
import psycopg2
import psycopg2.extras
from psycopg2 import pool as pg_pool

# 兼容旧引用（manage.py 提示文本中提到 it_helper.db），生产不再使用
DB_FILE = None

_pool: pg_pool.SimpleConnectionPool | None = None


def _dsn_kwargs() -> dict:
    """从环境变量读 PG 连接参数。"""
    return {
        "host": os.getenv("PG_HOST", "localhost"),
        "port": int(os.getenv("PG_PORT", "5432")),
        "dbname": os.getenv("PG_DB", "it_helper"),
        "user": os.getenv("PG_USER", "it_helper"),
        "password": os.getenv("PG_PASSWORD", ""),
    }


def _get_pool() -> pg_pool.SimpleConnectionPool:
    """懒初始化连接池。"""
    global _pool
    if _pool is None:
        _pool = pg_pool.SimpleConnectionPool(
            int(os.getenv("PG_POOL_MIN", "1")),
            int(os.getenv("PG_POOL_MAX", "10")),
            **_dsn_kwargs(),
        )
    return _pool


@contextmanager
def get_conn():
    """连接上下文管理器，从池借/还，自动 commit/rollback。

    返回的连接其 cursor 默认使用 RealDictCursor（行像 dict，兼容旧 sqlite3.Row 用法）。
    """
    p = _get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


def _dict_cursor(conn):
    """返回 RealDictCursor。统一行字典化。"""
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


def init_db():
    """建表 + 预置默认管理员。幂等，可重复调用。"""
    with get_conn() as conn:
        c = conn.cursor()
        # 用户表
        c.execute(
            """CREATE TABLE IF NOT EXISTS users (
                id BIGSERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('employee','it_admin')),
                display_name TEXT,
                must_change_pwd INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )"""
        )
        # 工单表
        c.execute(
            """CREATE TABLE IF NOT EXISTS tickets (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                issue TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL
            )"""
        )
        # 审计日志表
        c.execute(
            """CREATE TABLE IF NOT EXISTS audit_logs (
                id BIGSERIAL PRIMARY KEY,
                ts TEXT NOT NULL,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                tool TEXT NOT NULL,
                args TEXT,
                result TEXT,
                success INTEGER NOT NULL
            )"""
        )
        # 对话历史表
        c.execute(
            """CREATE TABLE IF NOT EXISTS conversations (
                id BIGSERIAL PRIMARY KEY,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts TEXT NOT NULL
            )"""
        )
        # 索引（覆盖现有按 username / 时间倒序查询）
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_username ON conversations(username)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_logs(ts DESC)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_tickets_username ON tickets(username)"
        )
        # 预置默认管理员
        c.execute("SELECT 1 FROM users WHERE username = %s", ("admin",))
        if c.fetchone() is None:
            pwd_hash = bcrypt.hashpw(
                b"admin123", bcrypt.gensalt()
            ).decode("utf-8")
            c.execute(
                "INSERT INTO users(username, password_hash, role, display_name, "
                "must_change_pwd, created_at) VALUES (%s,%s,%s,%s,%s,%s)",
                (
                    "admin",
                    pwd_hash,
                    "it_admin",
                    "系统管理员",
                    1,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )


# ---------- 用户/认证 ----------

def verify_user(username: str, password: str):
    """校验账号密码。成功返回 dict（含 role），失败返回 None。"""
    with get_conn() as conn:
        cur = _dict_cursor(conn)
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        row = cur.fetchone()
    if row is None:
        return None
    if not bcrypt.checkpw(
        password.encode("utf-8"), row["password_hash"].encode("utf-8")
    ):
        return None
    return dict(row)


def get_user(username: str):
    """按用户名查用户，返回 dict 或 None。"""
    with get_conn() as conn:
        cur = _dict_cursor(conn)
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        row = cur.fetchone()
    return dict(row) if row else None


def add_user(username: str, password: str, role: str, display_name: str = "") -> str:
    """新增用户。成功返回提示串，冲突/失败返回错误串。"""
    if role not in ("employee", "it_admin"):
        return f"非法角色：{role}"
    pwd_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users(username, password_hash, role, display_name, "
                "created_at) VALUES (%s,%s,%s,%s,%s)",
                (
                    username,
                    pwd_hash,
                    role,
                    display_name or username,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
    except psycopg2.IntegrityError:
        return f"用户已存在：{username}"
    return f"已添加用户 {username}（{role}）"


def change_password(username: str, new_password: str) -> str:
    """修改密码并清除 must_change_pwd 标记。"""
    pwd_hash = bcrypt.hashpw(
        new_password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET password_hash=%s, must_change_pwd=0 WHERE username=%s",
            (pwd_hash, username),
        )
        if cur.rowcount == 0:
            return f"用户不存在：{username}"
    return f"已修改 {username} 的密码"


def list_users():
    """列出全部用户（不含密码哈希）。"""
    with get_conn() as conn:
        cur = _dict_cursor(conn)
        cur.execute(
            "SELECT username, role, display_name, created_at FROM users ORDER BY id"
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ---------- 审计日志 ----------

def add_audit_log(
    username: str, role: str, tool: str, args: str, result: str, success: bool
):
    """写一条工具调用审计日志。"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO audit_logs(ts, username, role, tool, args, result, success) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (
                datetime.now().isoformat(timespec="seconds"),
                username,
                role,
                tool,
                args,
                result,
                1 if success else 0,
            ),
        )


def list_audit_logs(limit: int = 50):
    """查最近 N 条审计日志。"""
    with get_conn() as conn:
        cur = _dict_cursor(conn)
        cur.execute(
            "SELECT ts, username, role, tool, args, result, success "
            "FROM audit_logs ORDER BY id DESC LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ---------- 工单 ----------

def create_ticket_db(username: str, issue: str) -> str:
    """写一条工单，返回工单号。"""
    ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d')}-{datetime.now().strftime('%H%M%S')}"
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO tickets(id, username, issue, status, created_at) "
            "VALUES (%s,%s,%s,%s,%s)",
            (
                ticket_id,
                username,
                issue,
                "open",
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
    return ticket_id


def list_tickets(username: str | None = None, limit: int = 50):
    """查工单。username=None 查全部，否则只查该用户的。"""
    with get_conn() as conn:
        cur = _dict_cursor(conn)
        if username is None:
            cur.execute(
                "SELECT id, username, issue, status, created_at FROM tickets "
                "ORDER BY id DESC LIMIT %s",
                (limit,),
            )
        else:
            cur.execute(
                "SELECT id, username, issue, status, created_at FROM tickets "
                "WHERE username=%s ORDER BY id DESC LIMIT %s",
                (username, limit),
            )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ---------- 对话历史 ----------

def add_conversation(username: str, role: str, content: str):
    """写一条对话记录（用户提问/助手回答）。"""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO conversations(username, role, content, ts) "
            "VALUES (%s,%s,%s,%s)",
            (
                username,
                role,
                content,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )


def list_conversations(username: str | None = None, limit: int = 200):
    """查最近 N 条对话。username=None 查全部，否则只查该用户的。"""
    with get_conn() as conn:
        cur = _dict_cursor(conn)
        if username is None:
            cur.execute(
                "SELECT ts, username, role, content FROM conversations "
                "ORDER BY id DESC LIMIT %s",
                (limit,),
            )
        else:
            cur.execute(
                "SELECT ts, username, role, content FROM conversations "
                "WHERE username=%s ORDER BY id DESC LIMIT %s",
                (username, limit),
            )
        rows = cur.fetchall()
    return [dict(r) for r in rows]
