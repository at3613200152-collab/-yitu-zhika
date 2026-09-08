"""内测 - 训练数据采集与溯源层（方案 §3/§4/§5/§6/§7）。

原则：
- 原始预测与后续标签分开保存；用户修正不覆盖原始预测。
- 未知值一律用 None 并可在 unknown_notes 注明原因，绝不把缺失热量/重量写成 0。
- 标签分层：label_source ∈ {model_only, user_estimate, measured, reference}。
  只有 measured/reference 且通过审核才进入监督训练候选。
- 训练授权独立：training_consent 默认 False；未同意不进训练数据池。
- 图片私有存储：所有上传图片写入 data/annotations/（gitignore），不入 GitHub。
"""
import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "collection_v1.sqlite3"
ANNOT_DIR = ROOT / "data" / "annotations"          # 私有存储，gitignore
ANNOT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_SOURCES = {"model_only", "user_estimate", "measured", "reference"}
MASS_BASIS = {"raw", "as_served", "unknown"}
VALID_QUALITY = {"confirm_only", "directional", "manual_typed", "skipped"}


def _conn():
    c = sqlite3.connect(str(DB), check_same_thread=False, timeout=10.0)
    c.row_factory = sqlite3.Row
    return c


def init():
    c = _conn()
    c.execute("""
        CREATE TABLE IF NOT EXISTS records_v1 (
            record_id TEXT PRIMARY KEY,
            participant_id TEXT,
            meal_id TEXT,
            capture_session_id TEXT,
            dish_uuid TEXT,
            image_hash TEXT,
            model_version TEXT,
            original_prediction TEXT,       -- JSON 快照（原始预测，不被覆盖）
            label_value TEXT,               -- JSON：用户/实测标签
            label_source TEXT,              -- model_only|user_estimate|measured|reference
            mass_basis TEXT,                -- raw|as_served|unknown
            tare_status INTEGER,            -- 0/1/NULL
            cooking_method TEXT,            -- 用户声明，未知为 NULL
            ingredients TEXT,               -- 可选，JSON 列表
            quality TEXT,
            review_status TEXT,             -- pending|reviewed|excluded
            review_reason TEXT,
            training_consent INTEGER,       -- 0/1，默认 0
            consent_version TEXT,
            consent_time TEXT,
            dataset_split TEXT,
            client_ip TEXT,
            created_at TEXT NOT NULL
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_rec_dish ON records_v1(dish_uuid)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_rec_session ON records_v1(capture_session_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_rec_consent ON records_v1(training_consent)")

    c.execute("""
        CREATE TABLE IF NOT EXISTS consent_v1 (
            id TEXT PRIMARY KEY,
            participant_id TEXT,
            consent_version TEXT,
            granted INTEGER,
            consent_time TEXT,
            source TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS annotations_v1 (
            annotation_id TEXT PRIMARY KEY,
            record_id TEXT,
            capture_session_id TEXT,
            dish_uuid TEXT,
            image_path TEXT,                -- 私有存储路径
            image_hash TEXT,
            label_name TEXT,
            ingredients TEXT,
            measured_weight REAL,
            weight_unit TEXT,
            mass_basis TEXT,
            tare_status INTEGER,
            cooking_method TEXT,
            notes TEXT,
            participant_id TEXT,
            training_consent INTEGER,
            review_status TEXT,             -- pending|reviewed|excluded
            created_at TEXT NOT NULL
        )
    """)
    # 迁移：为已存在的表补 review_status
    cols = [r[1] for r in c.execute("PRAGMA table_info(annotations_v1)")]
    if "review_status" not in cols:
        c.execute("ALTER TABLE annotations_v1 ADD COLUMN review_status TEXT")
    c.commit()
    c.close()


init()


def ensure_consent_granted(participant_id, consent_version, source):
    """记录某参与者的训练授权（以最新一次为准）。返回是否同意。"""
    c = _conn()
    c.execute("INSERT INTO consent_v1 (id, participant_id, consent_version, granted, consent_time, source) VALUES (?,?,?,?,?,?)",
              (str(uuid.uuid4()), participant_id, consent_version, 1, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), source))
    c.commit()
    c.close()
    return True


def save_record(payload, client_ip):
    """保存一条确认的记录（含授权 + 溯源 + 标签分层）。返回 record_id。"""
    record_id = str(uuid.uuid4())
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    dish_uuid = payload.get("dish_uuid")
    if not dish_uuid:
        raise ValueError("缺少 dish_uuid")

    quality = payload.get("quality")
    if quality not in VALID_QUALITY:
        raise ValueError(f"quality 必须是 {'|'.join(sorted(VALID_QUALITY))}，得到 {quality}")

    # 原始预测快照（不被用户修正覆盖）
    original_prediction = {
        "model_version": payload.get("model_version"),
        "category_name": payload.get("model_category"),
        "calories": payload.get("model_calories"),
        "weight": payload.get("model_weight"),
        "category_probs": payload.get("model_category_probs"),
        "category_prob": payload.get("model_category_prob"),
    }
    # 标签值（用户填写的；未提供为 None，绝不填 0）
    label_value = {
        "corrected_calories": payload.get("corrected_calories"),
        "corrected_weight": payload.get("corrected_weight"),
        "corrected_category": payload.get("corrected_category"),
        "measured_weight": payload.get("measured_weight"),
    }
    # 标签来源：有实测重量 → measured；仅有用户填写 → user_estimate；只有确认 → model_only
    if payload.get("measured_weight") is not None:
        label_source = "measured"
    elif any(v is not None for v in (payload.get("corrected_calories"), payload.get("corrected_weight"), payload.get("corrected_category"))):
        label_source = "user_estimate"
    else:
        label_source = "model_only"

    consent = bool(payload.get("training_consent") and payload.get("consent_version"))
    consent_version = payload.get("consent_version") if consent else None
    consent_time = now if consent else None

    c = _conn()
    c.execute("""
        INSERT INTO records_v1
        (record_id, participant_id, meal_id, capture_session_id, dish_uuid, image_hash,
         model_version, original_prediction, label_value, label_source, mass_basis,
         tare_status, cooking_method, ingredients, quality, review_status, review_reason,
         training_consent, consent_version, consent_time, dataset_split, client_ip, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        record_id,
        payload.get("participant_id"),
        payload.get("meal_id"),
        payload.get("capture_session_id"),
        dish_uuid,
        payload.get("image_hash"),
        payload.get("model_version"),
        json.dumps(original_prediction, ensure_ascii=False),
        json.dumps(label_value, ensure_ascii=False),
        label_source,
        payload.get("mass_basis") if payload.get("mass_basis") in MASS_BASIS else None,
        payload.get("tare_status"),
        payload.get("cooking_method"),
        json.dumps(payload.get("ingredients"), ensure_ascii=False) if payload.get("ingredients") else None,
        quality,
        "pending",
        None,
        1 if consent else 0,
        consent_version,
        consent_time,
        payload.get("dataset_split"),
        client_ip,
        now,
    ))
    c.commit()
    c.close()

    if consent and payload.get("participant_id"):
        ensure_consent_granted(payload["participant_id"], consent_version, source="record")

    return record_id, label_source, consent


def save_volunteer_annotation(fields, image_bytes, client_ip):
    """自愿标注：菜名/食材/实测重量/烹饪方式 + 可选第二角度照片。返回 annotation_id 与审核状态。"""
    annotation_id = str(uuid.uuid4())
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    image_path = None
    image_hash = None
    if image_bytes and len(image_bytes) > 0:
        digest = hashlib.sha256(image_bytes).hexdigest()
        image_hash = digest
        ext = ".jpg"
        image_path = str(ANNOT_DIR / f"{annotation_id}{ext}")
        try:
            from io import BytesIO
            from PIL import Image
            # 转换为 RGB 再存 JPEG（丢弃 alpha/EXIF，隐私与兼容）
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            img.save(image_path, "JPEG", quality=85, optimize=True)
        except Exception:
            image_path = None
            image_hash = None

    c = _conn()
    c.execute("""
        INSERT INTO annotations_v1
        (annotation_id, record_id, capture_session_id, dish_uuid, image_path, image_hash,
         label_name, ingredients, measured_weight, weight_unit, mass_basis, tare_status,
         cooking_method, notes, participant_id, training_consent, review_status, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        annotation_id,
        fields.get("record_id"),
        fields.get("capture_session_id"),
        fields.get("dish_uuid"),
        image_path,
        image_hash,
        fields.get("label_name"),
        json.dumps(fields.get("ingredients"), ensure_ascii=False) if fields.get("ingredients") else None,
        fields.get("measured_weight"),
        fields.get("weight_unit"),
        fields.get("mass_basis") if fields.get("mass_basis") in MASS_BASIS else None,
        fields.get("tare_status"),
        fields.get("cooking_method"),
        fields.get("notes"),
        fields.get("participant_id"),
        1 if (fields.get("training_consent") and fields.get("consent_version")) else 0,
        "pending",
        now,
    ))
    c.commit()
    c.close()

    if fields.get("training_consent") and fields.get("consent_version") and fields.get("participant_id"):
        ensure_consent_granted(fields["participant_id"], fields["consent_version"], source="annotation")

    return annotation_id, "pending"
