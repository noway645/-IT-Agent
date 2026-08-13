"""企业 IT 助手 - P2 能力

基于 GLM-4.5-air + RAG + 工具调用 + 多轮记忆 + 账号认证。
P2 新增：首次登录强制改密 / 多 Tab 界面 / 速率限制 / 对话历史落库。
"""
import os
import re
import time
from collections import defaultdict, deque

import fastapi
import gradio as gr
import uvicorn
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.globals import set_verbose
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import db
import tools

# 加载 .env 中的 ZHIPUAI_API_KEY / LANGCHAIN_VERBOSE
load_dotenv()

# 启动时初始化数据库（建表 + 预置 admin）
db.init_db()

# verbose 模式：环境变量控制
_verbose = os.getenv("LANGCHAIN_VERBOSE", "false").lower() == "true"
if _verbose:
    set_verbose(True)

# 速率限制：每用户每分钟 N 次（滑动窗口）
_RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))
_rate_windows: dict[str, deque] = defaultdict(deque)

# 输入校验上限（防 prompt 注入炸 token）
MAX_MESSAGE_LEN = 2000  # 单条用户消息字符数上限


def _check_rate(username: str) -> str:
    """速率限制检查。返回空串说明允许，否则返回错误提示。"""
    now = time.time()
    dq = _rate_windows[username]
    while dq and dq[0] < now - 60:
        dq.popleft()
    if len(dq) >= _RATE_LIMIT:
        return f"操作过于频繁：每用户每分钟最多 {_RATE_LIMIT} 次，请稍后重试。"
    dq.append(now)
    return ""


# 初始化 GLM-4.5-air
llm = ChatZhipuAI(model="glm-4.5-air", temperature=0.3)

# Agent 系统提示词模板：{username}/{role} 由当前登录身份填充
SYSTEM_PROMPT_TEMPLATE = (
    "你是星辰科技 IT 部门的智能助手，负责帮助员工解决日常 IT 问题。\n"
    "当前登录员工用户名：{username}（角色：{role}）。该用户名即提问者的身份，"
    "员工说'我''我的'时指的就是 {username}。若员工询问'我是谁'，直接告知其用户名与角色。\n"
    "处理原则：\n"
    "1. 员工询问操作步骤（如 WiFi/VPN/邮箱/打印机设置）时，调用 search_knowledge_base 检索知识库后回答。\n"
    "2. 员工要求重置密码时，调用 reset_password。若员工说'重置我的密码'，username 参数填 {username}。"
    "若员工用'也''同样''顺便'等词追问，需结合上文判断是否要继续执行相同操作。\n"
    "3. 问题无法直接解决、需要人工跟进时，调用 create_ticket 创建工单。\n"
    "4. 回答用中文，分步骤、清晰、可操作。不要编造知识库中没有的参数。\n"
    "5. 调用工具后，用自然语言向员工复述结果。\n"
    "6. 每个工具最多调用一次，调用后直接给出最终回答，不要重复调用。"
)


def auth(username: str, password: str) -> bool:
    """Gradio 登录校验。"""
    return db.verify_user(username, password) is not None


def _clean_output(text: str) -> str:
    """清理 GLM reasoning 模型泄漏到输出的 <think> 标签；空内容兜底。"""
    if not text or not text.strip():
        return "抱歉，处理时未收到有效回复，请重试或联系 IT 服务台（8000）。"
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.replace("</think>", "").strip()


def _history_to_messages(history: list) -> list:
    """把 Gradio history 转 LangChain 消息列表。"""
    out = []
    for item in history:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out


def _run_agent_stream(message: str, history: list, username: str, role: str):
    """手动 Agent 循环（generator，yield 累积字符串）。

    第一轮非流式（需拿完整 tool_calls）；如有工具调用，执行后第二轮用
    llm.stream 逐字 yield 最终总结。调用方负责身份/速率/输入校验。
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(username=username, role=role)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(_history_to_messages(history))
    messages.append(HumanMessage(content=message))
    allowed_tools = tools.get_tools_for_role(role)
    llm_with_tools = llm.bind_tools(allowed_tools)
    tool_map = {t.name: t for t in allowed_tools}

    # 第一轮：模型决策是否调用工具（非流式，需完整 tool_calls）
    try:
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        if _verbose:
            print(f"[agent step 1] LLM 调用失败: {e}")
        yield f"抱歉，AI 服务暂时不可用，请稍后重试或联系 IT 服务台（8000）。错误：{e}"
        return
    if _verbose:
        print(f"[agent step 1] tool_calls={response.tool_calls}")
    if not response.tool_calls:
        yield _clean_output(response.content)
        return

    # 执行所有工具调用
    messages.append(response)
    progress = "🔧 正在调用工具处理您的请求，请稍候...\n\n"
    yield progress
    for tc in response.tool_calls:
        tool = tool_map.get(tc["name"])
        audit_ok = False
        if tool is None:
            result = f"错误：未知工具 {tc['name']}"
            checked_args = tc["args"]
        else:
            ok, reason, checked_args = tools.check_tool_permission(
                tc["name"], tc["args"], username, role
            )
            if not ok:
                result = reason
                if _verbose:
                    print(f"[perm {tc['name']}] 拒绝: {reason}")
            else:
                # 参数基础校验：截断超长参数 / 拒绝过多参数
                ok2, reason2, checked_args = tools.validate_tool_args(
                    tc["name"], checked_args
                )
                if not ok2:
                    result = reason2
                    if _verbose:
                        print(f"[validate {tc['name']}] 拒绝: {reason2}")
                else:
                    audit_ok = True
                    try:
                        result = tool.invoke(checked_args)
                    except Exception as e:
                        audit_ok = False
                        result = f"工具 {tc['name']} 执行失败：{e}。请稍后重试或改用其他方式。"
                        if _verbose:
                            print(f"[tool {tc['name']}] 异常: {e}")
        try:
            db.add_audit_log(
                username, role, tc["name"],
                str(checked_args), str(result)[:500], audit_ok,
            )
        except Exception as e:
            if _verbose:
                print(f"[audit] 写入失败: {e}")
        if _verbose:
            print(f"[tool {tc['name']}] args={checked_args} -> {result}")
        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    # 第二轮：强制总结（流式逐字）
    try:
        for chunk in llm.stream(messages):
            if chunk.content:
                progress += chunk.content
                yield progress
    except Exception as e:
        if _verbose:
            print(f"[agent final] LLM 流式调用失败: {e}")
        yield progress + (
            f"\n\n⚠️ 工具已执行，但总结回复时失败。请稍后重试或联系 IT 服务台（8000）。错误：{e}"
        )
        return
    # 最终清理（去 <think> 等泄漏标签）
    yield _clean_output(progress)


def respond(message: str, history: list, request: gr.Request):
    """ChatInterface 回调（generator）：校验 + 对话落库 + 流式输出。"""
    username = request.username or "匿名"
    user = db.get_user(username)
    if user is None:
        yield "错误：用户信息无效，请重新登录。"
        return
    role = user["role"]

    # 强制改密拦截
    if user["must_change_pwd"]:
        yield "⚠️ 您需要先修改初始密码才能使用助手。请切换到『修改密码』标签页完成修改。"
        return

    # 输入校验：消息长度上限，防 prompt 注入炸 token
    if not message or not message.strip():
        yield "请输入您的问题。"
        return
    if len(message) > MAX_MESSAGE_LEN:
        yield (
            f"⚠️ 输入过长（{len(message)} 字符），上限 {MAX_MESSAGE_LEN} 字符。"
            f"请精简问题后重试，或拆分为多条消息。"
        )
        return

    # 速率限制
    rate_err = _check_rate(username)
    if rate_err:
        yield rate_err
        return

    # 对话落库（用户问）
    try:
        db.add_conversation(username, role, f"USER: {message}")
    except Exception as e:
        if _verbose:
            print(f"[conv write] 失败(ask): {e}")

    # 流式输出 + 收集完整回答用于落库
    full_answer = ""
    for partial in _run_agent_stream(message, history, username, role):
        full_answer = partial
        yield partial

    # 对话落库（助手答）
    try:
        db.add_conversation(username, role, f"ASSISTANT: {full_answer}")
    except Exception as e:
        if _verbose:
            print(f"[conv write] 失败(ans): {e}")


# ---------- 修改密码 Tab ----------

def handle_change_pw(
    old_pw: str, new_pw: str, new_pw2: str, request: gr.Request
) -> str:
    """校验旧密码 → 两次新密码一致 → 长度≥8 → 调用 db.change_password。"""
    username = request.username
    if not username:
        return "❌ 未登录。"
    user = db.verify_user(username, old_pw)
    if user is None:
        return "❌ 旧密码错误。"
    if not new_pw or len(new_pw) < 8:
        return "❌ 新密码长度至少 8 位。"
    if new_pw != new_pw2:
        return "❌ 两次输入的新密码不一致。"
    if new_pw == old_pw:
        return "❌ 新密码不能与旧密码相同。"
    msg = db.change_password(username, new_pw)
    return f"✅ {msg}。请继续使用对话。"


# ---------- 管理员 Tab ----------

def admin_list_users(request: gr.Request) -> str:
    username = request.username or ""
    u = db.get_user(username)
    if u is None or u["role"] != "it_admin":
        return "❌ 无权限：仅 it_admin 可查看。"
    users = db.list_users()
    out = f"{'username':<16}{'role':<12}{'display_name':<16}created_at\n" + "-" * 56 + "\n"
    for r in users:
        out += f"{r['username']:<16}{r['role']:<12}{r['display_name']:<16}{r['created_at']}\n"
    return out


def admin_list_logs(limit: int, request: gr.Request) -> str:
    username = request.username or ""
    u = db.get_user(username)
    if u is None or u["role"] != "it_admin":
        return "❌ 无权限：仅 it_admin 可查看。"
    logs = db.list_audit_logs(limit)
    out = f"{'ts':<22}{'username':<12}{'role':<10}{'tool':<24}{'ok':<4}args\n" + "-" * 80 + "\n"
    for l in logs:
        out += f"{l['ts']:<22}{l['username']:<12}{l['role']:<10}{l['tool']:<24}{'Y' if l['success'] else 'N':<4}{str(l['args'])[:40]}\n"
    return out


def admin_list_tickets(limit: int, request: gr.Request) -> str:
    username = request.username or ""
    u = db.get_user(username)
    if u is None or u["role"] != "it_admin":
        return "❌ 无权限：仅 it_admin 可查看。"
    tickets = db.list_tickets(limit=limit)
    out = f"{'id':<22}{'username':<12}{'status':<8}{'created_at':<22}issue\n" + "-" * 80 + "\n"
    for t in tickets:
        issue = (t["issue"][:40] + "...") if len(t["issue"]) > 40 else t["issue"]
        out += f"{t['id']:<22}{t['username']:<12}{t['status']:<8}{t['created_at']:<22}{issue}\n"
    return out


# ---------- Blocks 多 Tab 组装 ----------

def build_ui():
    with gr.Blocks(title="企业 IT 助手") as demo:
        gr.Markdown("## 🏢 星辰科技 · IT 助手")

        with gr.Tabs() as tabs:
            with gr.Tab("💬 对话助手"):
                gr.ChatInterface(
                    fn=respond,
                    title="",
                    description="IT 问题智能解答 · 支持重置密码/提工单/查知识库",
                    examples=["我是谁？", "VPN 怎么装？", "帮我重置我的密码", "电脑蓝屏提工单"],
                )

            with gr.Tab("🔐 修改密码"):
                with gr.Row():
                    old_pw = gr.Textbox(label="旧密码", type="password")
                    new_pw = gr.Textbox(label="新密码（≥8 位）", type="password")
                    new_pw2 = gr.Textbox(label="再次输入新密码", type="password")
                pw_btn = gr.Button("修改密码", variant="primary")
                pw_result = gr.Markdown()
                pw_btn.click(
                    fn=handle_change_pw,
                    inputs=[old_pw, new_pw, new_pw2],
                    outputs=pw_result,
                )

            with gr.Tab("🛠 管理员"):
                gr.Markdown("仅 it_admin 角色可查看以下内容。")
                with gr.Row():
                    btn_users = gr.Button("📋 列出用户")
                    btn_logs = gr.Button("📝 查看审计日志")
                    btn_tickets = gr.Button("🗂 查看工单")
                    log_limit = gr.Slider(1, 200, 50, step=1, label="条数")
                admin_output = gr.Markdown()
                btn_users.click(fn=admin_list_users, outputs=admin_output)
                btn_logs.click(fn=admin_list_logs, inputs=log_limit, outputs=admin_output)
                btn_tickets.click(fn=admin_list_tickets, inputs=log_limit, outputs=admin_output)

    return demo


demo = build_ui()

# 自定义 FastAPI app：加 /health 端点 + 挂载 Gradio（保留 auth）
api = fastapi.FastAPI()


@api.get("/health")
def _health():
    """健康检查端点，供 Docker HEALTHCHECK / K8s 探针使用。"""
    return {"status": "ok"}


gr.mount_gradio_app(api, demo, path="/", auth=auth)


if __name__ == "__main__":
    uvicorn.run(
        api,
        host=os.getenv("SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("SERVER_PORT", "7860")),
    )
