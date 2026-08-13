"""管理 CLI - 用户/审计/工单管理

用法：
  python manage.py initdb                 # 初始化数据库
  python manage.py adduser <username> --role <employee|it_admin> [--name <显示名>]
  python manage.py changepw <username>
  python manage.py listusers
  python manage.py viewlogs [--limit 50]
  python manage.py viewtickets [--user <username>] [--limit 50]
"""
import argparse
import getpass
import sys

import db


def cmd_initdb(_args):
    db.init_db()
    print("数据库已初始化（it_helper.db）")


def cmd_adduser(args):
    pwd = getpass.getpass(f"为 {args.username} 设置密码: ")
    if not pwd:
        print("密码不能为空")
        sys.exit(1)
    msg = db.add_user(args.username, pwd, args.role, args.name)
    print(msg)


def cmd_changepw(args):
    pwd = getpass.getpass(f"为 {args.username} 设置新密码: ")
    if not pwd:
        print("密码不能为空")
        sys.exit(1)
    print(db.change_password(args.username, pwd))


def cmd_listusers(_args):
    users = db.list_users()
    if not users:
        print("（无用户）")
        return
    print(f"{'username':<16}{'role':<12}{'display_name':<16}created_at")
    print("-" * 60)
    for u in users:
        print(
            f"{u['username']:<16}{u['role']:<12}{u['display_name']:<16}{u['created_at']}"
        )


def cmd_viewlogs(args):
    logs = db.list_audit_logs(args.limit)
    if not logs:
        print("（无审计日志）")
        return
    print(f"{'ts':<22}{'username':<12}{'role':<12}{'tool':<24}{'ok':<4}args")
    print("-" * 80)
    for l in logs:
        print(
            f"{l['ts']:<22}{l['username']:<12}{l['role']:<12}{l['tool']:<24}"
            f"{'Y' if l['success'] else 'N':<4}{l['args']}"
        )


def cmd_viewtickets(args):
    tickets = db.list_tickets(args.user, args.limit)
    if not tickets:
        print("（无工单）")
        return
    print(f"{'id':<22}{'username':<12}{'status':<8}{'created_at':<22}issue")
    print("-" * 80)
    for t in tickets:
        issue = t["issue"][:40] + "..." if len(t["issue"]) > 40 else t["issue"]
        print(
            f"{t['id']:<22}{t['username']:<12}{t['status']:<8}{t['created_at']:<22}{issue}"
        )


def main():
    p = argparse.ArgumentParser(description="IT 助手管理 CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("initdb", help="初始化数据库").set_defaults(func=cmd_initdb)

    ap = sub.add_parser("adduser", help="添加用户")
    ap.add_argument("username")
    ap.add_argument("--role", required=True, choices=["employee", "it_admin"])
    ap.add_argument("--name", default="", help="显示名（可选）")
    ap.set_defaults(func=cmd_adduser)

    cp = sub.add_parser("changepw", help="修改密码")
    cp.add_argument("username")
    cp.set_defaults(func=cmd_changepw)

    sub.add_parser("listusers", help="列出用户").set_defaults(func=cmd_listusers)

    vl = sub.add_parser("viewlogs", help="查看审计日志")
    vl.add_argument("--limit", type=int, default=50)
    vl.set_defaults(func=cmd_viewlogs)

    vt = sub.add_parser("viewtickets", help="查看工单")
    vt.add_argument("--user", default=None)
    vt.add_argument("--limit", type=int, default=50)
    vt.set_defaults(func=cmd_viewtickets)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
