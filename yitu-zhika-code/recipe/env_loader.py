"""
一图知卡 - 环境变量加载器 (D3)

自动从 .env 文件加载环境变量，优先级：
系统环境变量 > .env 文件 > 默认值

用法：
    from env_loader import ensure_env, get_env, require_key

    ensure_env()  # 启动时调用一次
    key = require_key("DEEPSEEK_API_KEY")  # 缺失抛 RuntimeError
    model = get_env("DEEPSEEK_MODEL", default="deepseek-chat")
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional


def _find_dotenv(start: Path | None = None) -> Optional[Path]:
    """向上查找 .env 文件，最多 5 层。"""
    p = start or Path(__file__).resolve().parent
    for _ in range(5):
        candidate = p / ".env"
        if candidate.exists():
            return candidate
        if p.parent == p:
            break
        p = p.parent
    return None


def _parse_env_file(path: Path) -> dict:
    """简单解析 .env 文件，支持 # 注释和 KEY=VALUE。"""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        # 去掉引号
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            v = v[1:-1]
        out[k] = v
    return out


_loaded = False
_overrides: dict = {}


def load_env(force: bool = False) -> dict:
    """加载 .env，只对当前进程的环境变量做补充（不覆盖已存在的）。"""
    global _loaded
    if _loaded and not force:
        return dict(_overrides)
    dotenv = _find_dotenv()
    if dotenv is None:
        _loaded = True
        return {}
    parsed = _parse_env_file(dotenv)
    for k, v in parsed.items():
        if k not in os.environ:
            os.environ[k] = v
        _overrides[k] = v
    _loaded = True
    return dict(_overrides)


def ensure_env() -> None:
    """确保环境变量已加载。建议在应用启动时调用一次。"""
    load_env()


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """获取环境变量，自动加载 .env。"""
    if not _loaded:
        load_env()
    return os.environ.get(key, default)


def require_key(key: str, hint: str = "") -> str:
    """获取必需的环境变量，缺失抛 RuntimeError。"""
    val = get_env(key)
    if not val or val.startswith("sk-在此") or val == "":
        raise RuntimeError(
            f"Missing required env: {key}. "
            f"Please set it in .env file or as system env var. "
            f"{hint}".strip()
        )
    return val


if __name__ == "__main__":
    ensure_env()
    print("=== env_loader self-test ===")
    print(f".env overrides loaded: {len(_overrides)} keys")
    for k in _overrides:
        v = _overrides[k]
        # 隐藏敏感值
        if "KEY" in k or "TOKEN" in k or "SECRET" in k:
            masked = v[:6] + "***" if len(v) > 6 else "***"
            print(f"  {k} = {masked}")
        else:
            print(f"  {k} = {v}")
    # 测试 require_key
    try:
        key = require_key("DEEPSEEK_API_KEY")
        print(f"\nDEEPSEEK_API_KEY: {key[:6]}...{key[-4:]}")
    except RuntimeError as e:
        print(f"\nDEEPSEEK_API_KEY not configured: {e}")
