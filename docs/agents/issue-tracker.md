# Issue tracker: GitHub

Issues/specs 存放在 `at3613200152-collab/-yitu-zhika` 的 GitHub Issues，使用 `gh` CLI。

- 创建：`gh issue create --repo at3613200152-collab/-yitu-zhika --title "..." --body "..."`。
- 读取：`gh issue view <number> --repo at3613200152-collab/-yitu-zhika --comments`。
- 列出：`gh issue list --repo at3613200152-collab/-yitu-zhika --state open --json number,title,body,labels`。
- 评论：`gh issue comment <number> --repo at3613200152-collab/-yitu-zhika --body "..."`。
- 标签：`gh issue edit <number> --repo at3613200152-collab/-yitu-zhika --add-label "..."` / `--remove-label "..."`。
- 关闭：`gh issue close <number> --repo at3613200152-collab/-yitu-zhika --comment "..."`。

**PRs as a request surface: no.**

技能要求 publish to the issue tracker 时创建 issue；fetch ticket 时读取 issue 与评论。CLI 未登录时报告依赖，不索取聊天中的明文密钥，也不声称发布成功。
