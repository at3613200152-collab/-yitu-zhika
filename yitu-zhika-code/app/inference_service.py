"""独立模型推理服务（P0 安全加固版）。不依赖 Gradio，为微信小程序提供 REST API。

端点：
  GET  /health        — 健康检查 + 模型状态
  GET  /model-info    — 模型版本、精度、设备
  POST /predict       — 上传图片，返回 JSON 预测结果
  POST /feedback      — 提交用户反馈（手动修正）

P0 安全加固:
  - 鉴权：API key header (X-API-Key)，未授权拒绝
  - 上传限制：10MB 文件大小、仅允许 jpg/png
  - 限流：每 IP 30 req/min
  - 超时：30s 处理超时
  - 错误提示：统一 JSON 错误格式，不返回伪造成功
  - 日志：所有请求记录到 logs/inference_service.log

启动：
  python app/inference_service.py --port 8000
"""
import argparse
import hashlib
import io
import json
import logging
import os
import sqlite3
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from PIL import Image
from flask import Flask, request, jsonify

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

app = Flask(__name__)

# 全局状态
_pipeline = None
_pipeline_lock = threading.Lock()

# P0: 鉴权配置
API_KEY = os.environ.get("INFERENCE_API_KEY", "dev-key-change-in-prod")

# 商家端 / 审核端密钥（生产必须替换为独立强随机串，勿与用户端共用）
MERCHANT_API_KEY = os.environ.get("MERCHANT_API_KEY", "dev-merchant-key")
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "dev-admin-key")

# 微信登录（AppSecret 仅后端持有，见 app/auth.py）
WX_APPID = os.environ.get("WX_APPID", "wxcd827286e262c0ce")
WX_SECRET = os.environ.get("WX_SECRET", "")

# P0: 上线模型固定（见 docs/model_card.md）
# 不依赖测试集成绩挑模型，按 audit 完整性选定
ONLINE_MODEL_PATH = ROOT / "checkpoints" / "meal_rgb_official_v1" / "best.pt"
ONLINE_MODEL_SHA256 = "3f189a439fca3b79ecd820ae68605bcbfe61679def14f8dfa548c34261c89cf0"
ONLINE_MODEL_VERSION = "meal_rgb_official_v1"

# P0: 上传限制
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_MIMES = {"image/jpeg", "image/png", "image/jpg"}

# P0: 限流（每 IP 30 req/min）
RATE_LIMIT = 30
RATE_WINDOW = 60  # seconds
_rate_buckets = defaultdict(deque)
_rate_lock = threading.Lock()

# P0: 请求超时
REQUEST_TIMEOUT = 30  # seconds

# 日志
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "inference_service.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def get_pipeline():
    """懒加载推理管线，线程安全。"""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                # P1.1: 上线模型 SHA256 校验
                if not ONLINE_MODEL_PATH.exists():
                    raise RuntimeError(f"上线模型缺失：{ONLINE_MODEL_PATH}")
                import hashlib
                sha = hashlib.sha256(ONLINE_MODEL_PATH.read_bytes()).hexdigest()
                if sha != ONLINE_MODEL_SHA256:
                    raise RuntimeError(
                        f"上线模型 SHA256 不匹配："
                        f"expected={ONLINE_MODEL_SHA256[:16]}..., got={sha[:16]}..."
                    )
                logger.info(f"Model SHA256 verified: {sha[:16]}...")
                from app.experiment_pipeline import ExperimentPipeline
                _pipeline = ExperimentPipeline(device="cpu")
    return _pipeline


def check_api_key():
    """P0: 鉴权检查。未授权返回 False。"""
    key = request.headers.get("X-API-Key", "")
    if not key:
        return False
    return key == API_KEY


def check_rate_limit(client_ip: str) -> bool:
    """P0: 限流检查。每 IP 30 req/min。"""
    now = time.time()
    with _rate_lock:
        bucket = _rate_buckets[client_ip]
        # 移除过期记录
        while bucket and bucket[0] < now - RATE_WINDOW:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT:
            return False
        bucket.append(now)
        return True


def error_response(message: str, status_code: int, code: str = None):
    """P0: 统一错误响应格式。不返回伪造的成功。"""
    return jsonify({
        "status": "error",
        "code": code or f"ERR_{status_code}",
        "message": message,
    }), status_code


@app.before_request
def auth_and_limit():
    """P0: 所有请求前的鉴权 + 限流。"""
    path = request.path
    # health 端点开放（用于小程序探活）
    if path == "/health" and request.method == "GET":
        return None
    # 公开菜单 / 商家端 / 审核端：使用各自独立密钥，不走用户端 API key
    if path == "/public/menus" or path.startswith("/merchant/") or path == "/admin/review":
        return None

    # 鉴权
    if not check_api_key():
        logger.warning(f"Unauthorized from {request.remote_addr}")
        return error_response("未授权：缺少或错误的 API key", 401, "UNAUTHORIZED")

    # 限流
    client_ip = request.remote_addr or "unknown"
    if not check_rate_limit(client_ip):
        logger.warning(f"Rate limited: {client_ip}")
        return error_response(f"请求过于频繁（限 {RATE_LIMIT}/min）", 429, "RATE_LIMITED")

    return None


@app.route("/health")
def health():
    """健康检查。"""
    return jsonify({
        "status": "ok",
        "pipeline_loaded": _pipeline is not None,
    })


@app.route("/model-info")
def model_info():
    """模型版本和配置信息。"""
    try:
        pipe = get_pipeline()
        manifest = pipe.manifest
        return jsonify({
            "status": "ok",
            "models": {
                "rgb": "meal_rgb_official_v1",
                "nir": "meal_nir_official_v1",
                "external": "calorieclip_official_v1",
            },
            "categories": manifest.get("category_to_idx", {}),
            "precision": "FP32",
            "device": str(pipe.device),
        })
    except Exception as e:
        logger.error(f"model-info failed: {e}")
        return error_response(f"模型未就绪：{e}", 503, "MODEL_UNAVAILABLE")


@app.route("/predict", methods=["POST"])
def predict():
    """上传图片，返回预测结果。

    请求：
      Header: X-API-Key: <key>
      multipart/form-data, file 字段名 "image"

    响应：
      200 OK: {"status": "ok", "calories": ..., "weight": ..., "category_name": ...}
      400: 无图片 / 文件过大 / 格式不支持 / 图片损坏
      429: 限流
      503: 模型未就绪
    """
    # 1. 检查文件存在
    if "image" not in request.files:
        return error_response("缺少图片字段 image", 400, "NO_IMAGE")

    file = request.files["image"]

    # 2. 检查文件名
    if not file.filename:
        return error_response("空文件名", 400, "EMPTY_FILE")

    # 3. 检查 MIME 类型
    mime = file.content_type or ""
    if mime not in ALLOWED_MIMES:
        return error_response(
            f"不支持的图片格式：{mime}（仅允许 jpg/png）",
            400, "UNSUPPORTED_FORMAT"
        )

    # 4. 读取文件并检查大小
    file_data = file.read()
    if len(file_data) > MAX_FILE_SIZE:
        return error_response(
            f"文件过大：{len(file_data)/1024/1024:.1f}MB（上限 {MAX_FILE_SIZE/1024/1024}MB）",
            400, "FILE_TOO_LARGE"
        )

    # 5. 解码图片 + EXIF strip（隐私保护）
    try:
        image = Image.open(io.BytesIO(file_data))
        image.load()
        # P1: 去除 EXIF（GPS、相机序列号等）
        if hasattr(image, "_getexif") and image._getexif():
            data_pixels = list(image.getdata())
            image_no_exif = Image.new(image.mode, image.size)
            image_no_exif.putdata(data_pixels)
            image_no_exif.info = {}
            image = image_no_exif
            logger.info("EXIF stripped from upload")
    except Exception as e:
        return error_response(f"图片损坏或格式错误：{e}", 400, "INVALID_IMAGE")

    # 6. 食物前置过滤（ImageNet ResNet50）
    # 主多任务网络用 softmax 强行分类，对卡通/动漫/截图等域外数据会
    # 给出高置信度（甚至 1.0），必须在前端用一个覆盖类别更广的
    # ImageNet 预训练模型做"是否为食物"的二判
    try:
        from app.food_filter import get_food_filter
        food_filter = get_food_filter()
        is_food, ff_info = food_filter.is_food(image)
        if not is_food:
            logger.warning(
                f"reject non-food (ImageNet filter) from {request.remote_addr}: "
                f"top1={ff_info['top1_name']} "
                f"prob={ff_info['top1_prob']} "
                f"food_score={ff_info['food_score']} "
                f"top5={[(t['name'], t['prob']) for t in ff_info['top5']]}"
            )
            return jsonify({
                "status": "not_food",
                "message": "未检测到食物，请拍摄食物图片",
                "filter": "imagenet_resnet50",
                "top1_name": ff_info["top1_name"],
                "top1_prob": ff_info["top1_prob"],
                "food_score": ff_info["food_score"],
                "model_version": ONLINE_MODEL_VERSION,
            })
        logger.info(
            f"food filter pass from {request.remote_addr}: "
            f"top1={ff_info['top1_name']} "
            f"prob={ff_info['top1_prob']} "
            f"food_score={ff_info['food_score']}"
        )
    except Exception as e:
        # 过滤器加载失败不应阻断主流程，降级为仅用置信度阈值
        logger.error(f"food filter failed, fallback to confidence-only: {e}")

    # 7. 推理（主多任务网络）
    try:
        pipe = get_pipeline()
        result = pipe.predict(image)

        # 兜底：置信度阈值（在过滤器失效或漏过的低质量图片上仍能拦截）
        # 注意: 前置 ImageNet 过滤器已拦截非食物，这里阈值可放松到 0.25
        # 避免误拦真实食物（如肉类/甜点等置信度天然较低的类别）
        CONFIDENCE_THRESHOLD = 0.25
        prob = result.get("category_prob", 0.0)
        if not isinstance(prob, (int, float)):
            prob = 0.0
        if prob < CONFIDENCE_THRESHOLD:
            logger.warning(
                f"reject low-confidence from {request.remote_addr}: "
                f"prob={prob:.3f} < {CONFIDENCE_THRESHOLD}, "
                f"cat={result.get('category_name', '?')}"
            )
            return jsonify({
                "status": "not_food",
                "message": "未检测到食物，请拍摄食物图片",
                "filter": "confidence_threshold",
                "category_prob": round(float(prob), 3),
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "model_version": ONLINE_MODEL_VERSION,
            })

        # 过滤不可 JSON 序列化的字段
        safe = {k: v for k, v in result.items()
                if k != "nir_image" and isinstance(v, (int, float, str, bool, type(None)))}
        safe["category_idx"] = result.get("category_idx")
        safe["category_prob"] = result.get("category_prob")
        # 多类别置信度分布（前端"食物种类"提示）
        safe["category_probs"] = result.get("category_probs", [])
        safe["status"] = "ok"
        safe["model_version"] = ONLINE_MODEL_VERSION
        safe["model_sha256_prefix"] = ONLINE_MODEL_SHA256[:16]

        # 数值四舍五入到合理精度（避免 233.3663330078125 这种假精度）
        for k in ("calories", "weight", "category_prob"):
            if k in safe and isinstance(safe[k], float):
                safe[k] = round(safe[k], 1)

        logger.info(f"predict ok from {request.remote_addr}: "
                    f"cat={safe.get('category_name')}, "
                    f"kcal={safe.get('calories')}, "
                    f"prob={safe.get('category_prob')}")
        return jsonify(safe)

    except Exception as e:
        logger.error(f"predict failed: {e}", exc_info=True)
        # P0: 不返回伪造的成功，明确告知推理失败
        return error_response(
            f"推理失败：{e}",
            503,
            "INFERENCE_FAILED"
        )


# P1: SQLite feedback_v1 表（社区反馈设计文档第 6 节）
FEEDBACK_DB = ROOT / "data" / "feedback_v1.sqlite3"
FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)


def get_db():
    """获取 SQLite 连接，线程安全。"""
    conn = sqlite3.connect(str(FEEDBACK_DB), check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化 feedback_v1 表。"""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback_v1 (
            feedback_id TEXT PRIMARY KEY,
            dish_uuid TEXT NOT NULL,
            image_hash TEXT,
            prediction_id TEXT,
            user_grade TEXT,
            quality TEXT NOT NULL,
            corrected_calories REAL,
            corrected_weight REAL,
            corrected_category TEXT,
            model_version TEXT,
            model_calories REAL,
            model_weight REAL,
            model_category TEXT,
            client_ip TEXT,
            timestamp TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dish ON feedback_v1(dish_uuid)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_quality ON feedback_v1(quality)")
    conn.commit()
    conn.close()


# 应用启动时初始化 DB
init_db()


@app.route("/feedback", methods=["POST"])
def feedback():
    """提交用户反馈（升级版，支持三档点击 + quality 分级）。

    请求 JSON body：
      {
        "dish_uuid": "...",
        "image_hash": "...",
        "prediction_id": "...",
        "user_grade": "ok|over|under|skip|manual",
        "quality": "confirm_only|directional|manual_typed|skipped",
        "corrected_calories": 250,
        "corrected_weight": 180,
        "corrected_category": "vegetable",
        "model_version": "...",
        "model_calories": ...,
        "model_weight": ...,
        "model_category": ...
      }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        return error_response("无效 JSON", 400, "INVALID_JSON")

    dish_uuid = data.get("dish_uuid")
    quality = data.get("quality")
    if not dish_uuid:
        return error_response("缺少 dish_uuid", 400, "MISSING_DISH_UUID")
    if quality not in {"confirm_only", "directional", "manual_typed", "skipped"}:
        return error_response(
            f"quality 必须是 confirm_only|directional|manual_typed|skipped，得到 {quality}",
            400, "INVALID_QUALITY"
        )

    feedback_id = str(uuid.uuid4())
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 双写：SQLite + JSONL
    try:
        conn = get_db()
        conn.execute("""
            INSERT INTO feedback_v1
            (feedback_id, dish_uuid, image_hash, prediction_id, user_grade, quality,
             corrected_calories, corrected_weight, corrected_category,
             model_version, model_calories, model_weight, model_category,
             client_ip, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            feedback_id,
            dish_uuid,
            data.get("image_hash"),
            data.get("prediction_id"),
            data.get("user_grade"),
            quality,
            data.get("corrected_calories"),
            data.get("corrected_weight"),
            data.get("corrected_category"),
            data.get("model_version"),
            data.get("model_calories"),
            data.get("model_weight"),
            data.get("model_category"),
            request.remote_addr or "unknown",
            timestamp,
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"feedback DB save failed: {e}")
        return error_response(f"反馈保存失败：{e}", 500, "FEEDBACK_SAVE_FAILED")

    # JSONL 备份
    feedback_dir = ROOT / "data" / "user_feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    today = time.strftime("%Y-%m-%d", time.gmtime())
    feedback_file = feedback_dir / f"feedback_{today}.jsonl"
    entry = {
        "feedback_id": feedback_id,
        "timestamp": timestamp,
        "dish_uuid": dish_uuid,
        "image_hash": data.get("image_hash"),
        "prediction_id": data.get("prediction_id"),
        "user_grade": data.get("user_grade"),
        "quality": quality,
        "corrected_calories": data.get("corrected_calories"),
        "corrected_weight": data.get("corrected_weight"),
        "corrected_category": data.get("corrected_category"),
        "model_version": data.get("model_version"),
        "model_calories": data.get("model_calories"),
        "model_weight": data.get("model_weight"),
        "model_category": data.get("model_category"),
        "client_ip": request.remote_addr,
    }
    try:
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"feedback JSONL backup failed: {e}")

    logger.info(f"feedback saved: dish={dish_uuid}, quality={quality}")
    return jsonify({
        "status": "ok",
        "message": "反馈已记录",
        "feedback_id": feedback_id,
        "dish_uuid": dish_uuid,
        "quality": quality,
    })


@app.route("/record", methods=["POST"])
def record():
    """保存一条内测记录（含训练授权 + 溯源 + 标签分层）。见方案 §3/§5/§6。"""
    if not check_api_key():
        return error_response("未授权：缺少或错误的 API key", 401, "UNAUTHORIZED")
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        return error_response("无效 JSON", 400, "INVALID_JSON")
    try:
        from app import collection
        record_id, label_source, consent = collection.save_record(data, request.remote_addr or "unknown")
    except ValueError as e:
        return error_response(str(e), 400, "INVALID_RECORD")
    except Exception as e:
        logger.error(f"record save failed: {e}", exc_info=True)
        return error_response(f"记录保存失败：{e}", 500, "RECORD_SAVE_FAILED")
    return jsonify({
        "status": "ok", "record_id": record_id, "label_source": label_source,
        "training_consent": consent, "review_status": "pending",
    })


@app.route("/annotate", methods=["POST"])
def annotate():
    """自愿标注（可跳过）：菜名/食材/实测重量/烹饪方式 + 可选第二角度照片。见方案 §3.2。"""
    if not check_api_key():
        return error_response("未授权：缺少或错误的 API key", 401, "UNAUTHORIZED")
    fields = {}
    image_bytes = None
    if request.mimetype and request.mimetype.startswith("multipart/"):
        for k in ("record_id", "capture_session_id", "dish_uuid", "label_name", "ingredients",
                  "measured_weight", "weight_unit", "mass_basis", "tare_status", "cooking_method",
                  "notes", "participant_id", "training_consent", "consent_version"):
            if k in request.form:
                fields[k] = request.form.get(k)
        image_bytes = request.files["image"].read() if "image" in request.files else None
    else:
        try:
            data = request.get_json(force=True, silent=True) or {}
        except Exception:
            return error_response("无效 JSON", 400, "INVALID_JSON")
        fields = data
    for k in ("measured_weight", "tare_status"):
        if k in fields and fields[k] not in (None, ""):
            try:
                fields[k] = float(fields[k]) if k == "measured_weight" else int(float(fields[k]))
            except (ValueError, TypeError):
                fields[k] = None
    try:
        from app import collection
        ann_id, status = collection.save_volunteer_annotation(fields, image_bytes, request.remote_addr or "unknown")
    except Exception as e:
        logger.error(f"annotate failed: {e}", exc_info=True)
        return error_response(f"标注保存失败：{e}", 500, "ANNOTATE_FAILED")
    return jsonify({"status": "ok", "annotation_id": ann_id, "review_status": status})


@app.route("/consent", methods=["POST"])
def consent():
    """记录某参与者的训练授权版本（审计用）。"""
    if not check_api_key():
        return error_response("未授权：缺少或错误的 API key", 401, "UNAUTHORIZED")
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        return error_response("无效 JSON", 400, "INVALID_JSON")
    pid = data.get("participant_id")
    ver = data.get("consent_version")
    if not pid or not ver:
        return error_response("缺少 participant_id 或 consent_version", 400, "MISSING_CONSENT")
    try:
        from app import collection
        collection.ensure_consent_granted(pid, ver, source="explicit")
    except Exception as e:
        logger.error(f"consent save failed: {e}", exc_info=True)
        return error_response(f"授权保存失败：{e}", 500, "CONSENT_FAILED")
    return jsonify({"status": "ok", "message": "授权已记录", "participant_id": pid, "consent_version": ver})


@app.route("/weekly-plan", methods=["POST"])
def weekly_plan():
    """生成 7 天食谱（模板版，不依赖 DS API）。

    请求 JSON body：
      {
        "height_cm": 170,
        "weight_kg": 70,
        "age": 30,
        "gender": "male",
        "activity_level": "moderate",
        "goal": "maintain",
        "allergies": ["鸡蛋", "海鲜"]
      }
    """
    if not check_api_key():
        return error_response("未授权：缺少或错误的 API key", 401, "UNAUTHORIZED")

    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        return error_response("无效 JSON", 400, "INVALID_JSON")

    required = ["height_cm", "weight_kg", "age", "gender"]
    for k in required:
        if k not in data:
            return error_response(f"缺少参数：{k}", 400, "MISSING_PARAM")

    try:
        sys.path.insert(0, str(ROOT))
        from recipe.tdee_estimator import UserProfile, Gender, ActivityLevel, Goal
        from recipe.template_planner import TemplatePlanner
    except ImportError as e:
        return error_response(f"营养师模块加载失败：{e}", 500, "RECIPE_IMPORT_FAILED")

    gender_map = {"male": Gender.MALE, "female": Gender.FEMALE}
    activity_map = {
        "sedentary": ActivityLevel.SEDENTARY,
        "light": ActivityLevel.LIGHT,
        "moderate": ActivityLevel.MODERATE,
        "active": ActivityLevel.ACTIVE,
        "very_active": ActivityLevel.VERY_ACTIVE,
    }
    goal_map = {"lose": Goal.LOSE, "maintain": Goal.MAINTAIN, "gain": Goal.GAIN}

    gender = gender_map.get(str(data["gender"]).lower())
    if not gender:
        return error_response("gender 必须是 male 或 female", 400, "INVALID_GENDER")

    activity = activity_map.get(str(data.get("activity_level", "moderate")).lower(), ActivityLevel.MODERATE)
    goal = goal_map.get(str(data.get("goal", "maintain")).lower(), Goal.MAINTAIN)

    profile = UserProfile(
        height_cm=float(data["height_cm"]),
        weight_kg=float(data["weight_kg"]),
        age=int(data["age"]),
        gender=gender,
        activity_level=activity,
        goal=goal,
        allergies=data.get("allergies", []),
        preferences=data.get("preferences", []),
        special_population=bool(data.get("special_population", False)),
    )

    try:
        planner = TemplatePlanner()
        result = planner.generate(profile)
        logger.info(f"weekly-plan generated: status={result['status']}, days={len(result.get('daily_recipes', []))}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"weekly-plan failed: {e}", exc_info=True)
        return error_response(f"食谱生成失败：{e}", 500, "WEEKLY_PLAN_FAILED")


@app.route("/plan-from-menu", methods=["POST"])
def plan_from_menu():
    """从自定义/预设食物池生成饮食参考（用户或商家补全食物营养数据后调用）。

    请求 JSON body（在 /weekly-plan 档案字段基础上，二选一或都提供）：
      {
        ...profile 字段（height_cm/weight_kg/age/gender/activity_level/goal/allergies...）,
        "preset": "dumpling" | "merchant_demo",          # 可选：内置预设菜单
        "foods": [                                        # 可选：用户/商家补全的食物
          {"name":"猪肉白菜水饺","category":"grain","kcal_per_100g":222,
           "protein_per_100g":9.5,"carb_per_100g":28,"fat_per_100g":9,"default_grams":150},
          ...
        ]
      }
    """
    if not check_api_key():
        return error_response("未授权：缺少或错误的 API key", 401, "UNAUTHORIZED")
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        return error_response("无效 JSON", 400, "INVALID_JSON")

    required = ["height_cm", "weight_kg", "age", "gender"]
    for k in required:
        if k not in data:
            return error_response(f"缺少参数：{k}", 400, "MISSING_PARAM")

    try:
        sys.path.insert(0, str(ROOT))
        from recipe.tdee_estimator import UserProfile, Gender, ActivityLevel, Goal
        from recipe.custom_menu import (
            validate_food, foods_from_preset, plan_from_pool,
        )
    except ImportError as e:
        return error_response(f"营养师模块加载失败：{e}", 500, "RECIPE_IMPORT_FAILED")

    gender_map = {"male": Gender.MALE, "female": Gender.FEMALE}
    activity_map = {
        "sedentary": ActivityLevel.SEDENTARY, "light": ActivityLevel.LIGHT,
        "moderate": ActivityLevel.MODERATE, "active": ActivityLevel.ACTIVE,
        "very_active": ActivityLevel.VERY_ACTIVE,
    }
    goal_map = {"lose": Goal.LOSE, "maintain": Goal.MAINTAIN, "gain": Goal.GAIN}

    gender = gender_map.get(str(data["gender"]).lower())
    if not gender:
        return error_response("gender 必须是 male 或 female", 400, "INVALID_GENDER")
    activity = activity_map.get(str(data.get("activity_level", "moderate")).lower(), ActivityLevel.MODERATE)
    goal = goal_map.get(str(data.get("goal", "maintain")).lower(), Goal.MAINTAIN)

    profile = UserProfile(
        height_cm=float(data["height_cm"]),
        weight_kg=float(data["weight_kg"]),
        age=int(data["age"]),
        gender=gender,
        activity_level=activity,
        goal=goal,
        allergies=data.get("allergies", []),
        preferences=data.get("preferences", []),
        special_population=bool(data.get("special_population", False)),
    )

    # 组装食物池：商家菜单(menu_id) + 预设 + 用户补全
    foods = []
    menu_id = str(data.get("menu_id", "")).strip()
    if menu_id:
        from app import merchant
        products = merchant.get_menu_products(menu_id)
        if products is None:
            return error_response("菜单不存在或未通过审核", 400, "UNKNOWN_MENU")
        for p in products:
            raw = {
                "name": p["name"], "category": p.get("category"), "kcal_per_100g": p.get("kcal_per_100g"),
                "protein_per_100g": p.get("protein_per_100g"), "carb_per_100g": p.get("carb_per_100g"),
                "fat_per_100g": p.get("fat_per_100g"), "default_grams": p.get("default_grams"),
                "source_type": "merchant", "source_ref": menu_id,
            }
            item, err = validate_food(raw)
            if err:
                return error_response(f"菜单产品「{p['name']}」无效：{err}", 400, "INVALID_FOOD")
            foods.append(item)

    preset_name = str(data.get("preset", "")).strip()
    if preset_name:
        preset_foods, pname, pdesc = foods_from_preset(preset_name)
        if not preset_foods:
            return error_response(f"未知预设菜单：{preset_name}", 400, "UNKNOWN_PRESET")
        foods.extend(preset_foods)

    user_foods_raw = data.get("foods", [])
    if isinstance(user_foods_raw, dict):
        user_foods_raw = [user_foods_raw]
    if user_foods_raw:
        for idx, raw in enumerate(user_foods_raw):
            if not isinstance(raw, dict):
                return error_response(f"foods[{idx}] 必须是对象", 400, "INVALID_FOOD")
            item, err = validate_food(raw)
            if err:
                return error_response(f"foods[{idx}] {err}", 400, "INVALID_FOOD")
            foods.append(item)

    if not foods:
        return error_response("请提供 preset 或 foods 至少其一", 400, "EMPTY_FOOD_POOL")

    try:
        result = plan_from_pool(profile, foods)
        logger.info(
            f"plan-from-menu generated: status={result['status']}, "
            f"days={len(result.get('daily_recipes', []))}, pool={result.get('food_pool_count')}"
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"plan-from-menu failed: {e}", exc_info=True)
        return error_response(f"饮食安排生成失败：{e}", 500, "PLAN_FAILED")


@app.route("/merchant/menu", methods=["POST"])
def merchant_create_menu():
    """商家端：商家创建自己的菜单与产品（需 X-Merchant-Key）。审核前不对外展示。"""
    key = request.headers.get("X-Merchant-Key", "")
    if key != MERCHANT_API_KEY:
        return error_response("未授权：商家密钥无效", 401, "UNAUTHORIZED")
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        return error_response("无效 JSON", 400, "INVALID_JSON")
    merchant_name = str(data.get("merchant_name", "")).strip()
    menu_name = str(data.get("menu_name", "")).strip()
    products = data.get("products")
    if not merchant_name or not menu_name:
        return error_response("缺少 merchant_name 或 menu_name", 400, "MISSING_PARAM")
    if not isinstance(products, list) or not products:
        return error_response("products 需为非空数组", 400, "INVALID_PRODUCTS")
    for p in products:
        if not p.get("name") or p.get("kcal_per_100g") is None:
            return error_response("每个产品需有 name 和 kcal_per_100g", 400, "INVALID_PRODUCTS")
    try:
        from app import merchant
        menu_id = merchant.create_menu(merchant_name, menu_name, products, data.get("desc"))
    except Exception as e:
        logger.error(f"merchant create failed: {e}", exc_info=True)
        return error_response(f"创建菜单失败：{e}", 500, "MERCHANT_FAILED")
    return jsonify({"status": "ok", "menu_id": menu_id, "review_status": "pending"})


@app.route("/merchant/menus", methods=["GET"])
def merchant_list_menus():
    """商家查自己的菜单（需 X-Merchant-Key）。"""
    key = request.headers.get("X-Merchant-Key", "")
    if key != MERCHANT_API_KEY:
        return error_response("未授权：商家密钥无效", 401, "UNAUTHORIZED")
    try:
        from app import merchant
        merchant_name = request.args.get("merchant_name")
        menus = merchant.list_menus(merchant_name)
    except Exception as e:
        return error_response(f"查询失败：{e}", 500, "MERCHANT_FAILED")
    return jsonify({"status": "ok", "menus": menus})


@app.route("/admin/review", methods=["POST"])
def admin_review():
    """审核端：设置菜单审核状态（pending/approved/excluded）（需 X-Admin-Key）。"""
    key = request.headers.get("X-Admin-Key", "")
    if key != ADMIN_API_KEY:
        return error_response("未授权：审核密钥无效", 401, "UNAUTHORIZED")
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        return error_response("无效 JSON", 400, "INVALID_JSON")
    menu_id = data.get("menu_id")
    status = data.get("status")
    if not menu_id or not status:
        return error_response("缺少 menu_id/status", 400, "MISSING_PARAM")
    try:
        from app import merchant
        ok = merchant.set_review(menu_id, status)
    except ValueError as e:
        return error_response(str(e), 400, "INVALID_STATUS")
    except Exception as e:
        return error_response(f"审核失败：{e}", 500, "REVIEW_FAILED")
    return jsonify({"status": "ok", "menu_id": menu_id, "review_status": status, "updated": ok})


@app.route("/public/menus", methods=["GET"])
def public_menus():
    """用户侧公开菜单列表：仅返回已审核通过的商家菜单。"""
    try:
        from app import merchant
        menus = merchant.public_menus()
    except Exception as e:
        return error_response(f"查询失败：{e}", 500, "MERCHANT_FAILED")
    return jsonify({"status": "ok", "menus": menus})


@app.route("/auth/wx-login", methods=["POST"])
def wx_login():
    """微信登录：前端传 wx.login 的 code，后端换 openid 并返回去标识化 participant_id。"""
    if not check_api_key():
        return error_response("未授权：缺少或错误的 API key", 401, "UNAUTHORIZED")
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        return error_response("无效 JSON", 400, "INVALID_JSON")
    code = data.get("code")
    if not code:
        return error_response("缺少 code", 400, "MISSING_CODE")
    try:
        from app import auth
        pid, verified = auth.wx_login(code, WX_APPID, WX_SECRET)
    except ValueError as e:
        return error_response(str(e), 400, "WX_LOGIN_FAILED")
    except Exception as e:
        logger.error(f"wx-login error: {e}", exc_info=True)
        return error_response(f"微信登录失败：{e}", 500, "WX_LOGIN_ERROR")
    return jsonify({"status": "ok", "participant_id": pid, "wx_verified": verified})


def main():
    parser = argparse.ArgumentParser(description="yitu-zhika inference service")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    # 预加载管线
    print("Loading inference pipeline...")
    get_pipeline()
    print("Pipeline ready. Starting server...")

    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
