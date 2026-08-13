"""工具函数模块

模拟企业 IT 操作：重置密码、创建工单。
另含知识库检索工具，供 Agent 自主调用。
权限校验与参数校验函数供 app.py 调用。
"""
from langchain_core.tools import tool

import db
import rag

# 输入校验上限（防 prompt 注入炸 token / 工具入参超长）
MAX_TOOL_ARG_LEN = 1000  # 单个工具参数字符串最大长度
MAX_TOOL_ARGS = 8  # 单次工具调用参数个数上限

# 检索器懒加载（首次调用工具时初始化，避免启动即加载向量库）
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = rag.get_retriever()
    return _retriever


@tool
def reset_password(username: str) -> str:
    """重置指定员工的 OA/域账号密码（模拟操作，不产生真实副作用）。

    当员工明确要求重置自己或他人密码时调用。

    Args:
        username: 员工用户名，如 zhangsan 或 firstname.lastname
    """
    return (
        f"已为 {username} 重置密码（模拟）。"
        f"新密码已通过企业微信发送给本人，首次登录后请立即修改。"
    )


@tool
def create_ticket(issue_desc: str, username: str = "匿名员工") -> str:
    """创建一条 IT 工单记录。当问题无法直接解决、需要人工跟进时使用。

    Args:
        issue_desc: 问题描述
        username: 报障员工用户名，未提供时记为"匿名员工"
    """
    try:
        ticket_id = db.create_ticket_db(username, issue_desc)
    except Exception as e:
        return f"工单写入数据库失败：{e}。请联系 IT 服务台（8000）手动登记。"
    return f"工单已创建：{ticket_id}，状态 open。IT 服务台（8000）将跟进处理。"


@tool
def search_knowledge_base(query: str) -> str:
    """检索企业 IT 知识库，获取 WiFi/VPN/邮箱/打印机等操作指南。

    当员工询问"怎么设置""如何连接""安装步骤"等知识性问题时调用。

    Args:
        query: 检索关键词或完整问题
    """
    docs = _get_retriever().invoke(query)
    if not docs:
        return "知识库中未检索到相关内容。"
    return "\n\n---\n\n".join(d.page_content for d in docs)


# 供 Agent 使用的工具清单
ALL_TOOLS = [reset_password, create_ticket, search_knowledge_base]


def get_tools_for_role(role: str) -> list:
    """按角色返回可用工具。当前矩阵所有工具对 employee/admin 均可见，
    细粒度权限在工具执行段做参数级校验（见 app.py _run_agent）。
    未来新增管理类工具时在此过滤。
    """
    return ALL_TOOLS


def check_tool_permission(
    tool_name: str, args: dict, current_user: str, current_role: str
) -> tuple[bool, str, dict]:
    """工具参数级权限校验（第二层防御，防 LLM 构造非法参数）。

    返回 (是否允许, 拒绝原因/空, 校正后的 args)。
    - reset_password: employee 仅能重置自己，admin 任意
    - create_ticket: employee 强制 username=自己（防冒名），admin 保持原值
    - search_knowledge_base: 无限制
    """
    args = dict(args)  # 拷贝，不污染原参数
    if tool_name == "reset_password":
        target = args.get("username", "")
        if current_role != "it_admin" and target != current_user:
            return False, (
                f"无权限：普通员工仅可重置自己的密码（{current_user}），"
                f"重置他人密码请联系 IT 管理员。"
            ), args
    elif tool_name == "create_ticket":
        if current_role != "it_admin":
            args["username"] = current_user  # 强制归属当前用户
    return True, "", args


def validate_tool_args(tool_name: str, args: dict) -> tuple[bool, str, dict]:
    """工具参数基础校验（防 LLM 构造超长/过多参数炸 token 或注入）。

    返回 (是否合法, 拒绝原因/空, 截断后的 args)。
    - 参数个数上限 MAX_TOOL_ARGS
    - 单个字符串参数长度上限 MAX_TOOL_ARG_LEN（超长截断，不直接拒绝，保可用性）
    - 非 str/数字类型转 str 后再校验
    """
    args = dict(args)
    if len(args) > MAX_TOOL_ARGS:
        return False, f"参数过多（{len(args)}>{MAX_TOOL_ARGS}），疑似异常调用。", args
    for k, v in args.items():
        if v is None:
            continue
        s = v if isinstance(v, str) else str(v)
        if len(s) > MAX_TOOL_ARG_LEN:
            # 截断而非拒绝，避免 LLM 偶发生成长文本导致整轮失败
            args[k] = s[:MAX_TOOL_ARG_LEN]
    return True, "", args
