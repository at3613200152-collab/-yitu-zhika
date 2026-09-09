"""商家端 - 商家菜单/产品库与审核（训练数据采集方案之外的业务能力）。

背景：开发期把 "饺子餐(预设)/商家推广(预设)" 直接暴露给用户既难看又暴露测试数据。
正确模型：商家/运营后台创建自己的菜单与产品 → 审核通过 → 用户侧看到 "商家名·菜单名" 这类选项。

本期实现（后端可验证）：
- merchant_menus / merchant_products 两表，含溯源 + review_status(pending/approved/excluded)。
- POST /merchant/menu        创建菜单（需 X-Merchant-Key）
- GET  /merchant/menus        商家查自己的菜单（需 X-Merchant-Key）
- POST /admin/review          审核（需 X-Admin-Key）
- GET  /public/menus          公开列表：仅返回 approved 的菜单+产品（无需鉴权）
- plan-from-menu 支持 menu_id：从 approved 菜单产品生成饮食计划。

边界：产品营养数值由商家提供，属"用户声明/商家提供"层，未经核验前不进入训练真值；
仅在明确标注来源时才用于营养计算（区别于 nutrition_db 的核验数据）。
"""
import json
import sqlite3
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "merchant_v1.sqlite3"

VALID_REVIEW = {"pending", "approved", "excluded"}


def _conn():
    c = sqlite3.connect(str(DB), check_same_thread=False, timeout=10.0)
    c.row_factory = sqlite3.Row
    return c


def init():
    c = _conn()
    c.execute("""
        CREATE TABLE IF NOT EXISTS merchant_menus (
            menu_id TEXT PRIMARY KEY,
            merchant_name TEXT NOT NULL,
            menu_name TEXT NOT NULL,
            desc TEXT,
            review_status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS merchant_products (
            product_id TEXT PRIMARY KEY,
            menu_id TEXT NOT NULL,
            name TEXT NOT NULL,
            category TEXT,
            kcal_per_100g REAL,
            protein_per_100g REAL,
            carb_per_100g REAL,
            fat_per_100g REAL,
            default_grams REAL,
            source TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_mp_menu ON merchant_products(menu_id)")
    c.commit()
    c.close()


init()


def create_menu(merchant_name, menu_name, products, desc=None):
    menu_id = str(uuid.uuid4())
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    c = _conn()
    c.execute("INSERT INTO merchant_menus (menu_id, merchant_name, menu_name, desc, review_status, created_at) VALUES (?,?,?,?,?,?)",
              (menu_id, merchant_name, menu_name, desc, "pending", now))
    for p in products:
        c.execute("""
            INSERT INTO merchant_products
            (product_id, menu_id, name, category, kcal_per_100g, protein_per_100g,
             carb_per_100g, fat_per_100g, default_grams, source)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (str(uuid.uuid4()), menu_id, p["name"], p.get("category"), p.get("kcal_per_100g"),
              p.get("protein_per_100g"), p.get("carb_per_100g"), p.get("fat_per_100g"),
              p.get("default_grams"), p.get("source", "merchant")))
    c.commit()
    c.close()
    return menu_id


def list_menus(merchant_name=None):
    c = _conn()
    if merchant_name:
        rows = c.execute("SELECT * FROM merchant_menus WHERE merchant_name=?", (merchant_name,)).fetchall()
    else:
        rows = c.execute("SELECT * FROM merchant_menus").fetchall()
    c.close()
    return [dict(r) for r in rows]


def set_review(menu_id, status):
    if status not in VALID_REVIEW:
        raise ValueError(f"status 必须是 {'|'.join(VALID_REVIEW)}")
    c = _conn()
    cur = c.execute("UPDATE merchant_menus SET review_status=? WHERE menu_id=?", (status, menu_id))
    c.commit()
    c.close()
    return cur.rowcount > 0


def public_menus():
    """仅返回已审核通过的菜单（含产品，按来源标明）。"""
    c = _conn()
    menus = c.execute("SELECT * FROM merchant_menus WHERE review_status='approved' ORDER BY created_at DESC").fetchall()
    out = []
    for m in menus:
        products = c.execute("SELECT * FROM merchant_products WHERE menu_id=?", (m["menu_id"],)).fetchall()
        out.append({
            "menu_id": m["menu_id"],
            "merchant_name": m["merchant_name"],
            "menu_name": m["menu_name"],
            "desc": m["desc"],
            "products": [dict(p) for p in products],
        })
    c.close()
    return out


def get_menu_products(menu_id):
    """取某菜单的产品（仅 approved）。返回 product dict 列表或 None。"""
    c = _conn()
    menu = c.execute("SELECT * FROM merchant_menus WHERE menu_id=? AND review_status='approved'", (menu_id,)).fetchone()
    if not menu:
        c.close()
        return None
    products = c.execute("SELECT * FROM merchant_products WHERE menu_id=?", (menu_id,)).fetchall()
    c.close()
    return [dict(p) for p in products]
