"""微信登录：wx.login code → code2session → 去标识化 participant_id。

要点（对齐内测方案 §7 与 §5）：
- AppSecret 只在服务器端，绝不进小程序代码。
- 保存 openid → participant_id 映射（开放标识与参与者编号分离存储，仅存服务器私有库；不是匿名化保证）。
- 返回给客户端的只有 participant_id（去标识化），不回传 openid/session_key。
- 未配置 WX_SECRET 时降级：返回本地生成的去标识编号（wx_verified=False），便于未上线微信前先跑通记录流程。
"""
import hashlib
import json
import sqlite3
import time
import uuid
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "auth_v1.sqlite3"


def _conn():
    c = sqlite3.connect(str(DB), check_same_thread=False, timeout=10.0)
    c.row_factory = sqlite3.Row
    return c


def init():
    c = _conn()
    c.execute("""
        CREATE TABLE IF NOT EXISTS wx_users (
            openid TEXT PRIMARY KEY,
            participant_id TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    c.commit()
    c.close()


init()


def _participant_from_openid(openid):
    return "wx_" + hashlib.sha256(openid.encode("utf-8")).hexdigest()[:16]


def wx_login(code, appid, secret):
    """用 wx.login 的 code 换取 openid，并返回 (participant_id, wx_verified) 元组。"""
    if not appid or not secret:
        # 尚未配置微信密钥：降级，返回本地去标识编号（无法校验微信身份）
        return "u_" + uuid.uuid4().hex[:12], False

    params = urllib.parse.urlencode({
        "appid": appid,
        "secret": secret,
        "js_code": code,
        "grant_type": "authorization_code",
    })
    url = "https://api.weixin.qq.com/sns/jscode2session?" + params
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise ValueError("微信登录接口调用失败：" + str(e))

    if data.get("errcode", 0) != 0:
        raise ValueError("微信登录失败：%s(%s)" % (data.get("errmsg"), data.get("errcode")))

    openid = data["openid"]
    pid = _participant_from_openid(openid)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    c = _conn()
    c.execute("INSERT OR IGNORE INTO wx_users (openid, participant_id, created_at) VALUES (?,?,?)",
              (openid, pid, now))
    c.commit()
    c.close()
    return pid, True
