"""
Flask 后端服务
==============
启动方式:
    python app/server.py
或者:
    python -m flask --app app.server run --port 5000
"""

import os
import io
import sys
import base64
import glob
import logging
import traceback
from typing import Optional

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

# 让 Python 能 import 仓库根目录的模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from inference.estimator import FoodCalorieEstimator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
)
log = logging.getLogger("app.server")

app = Flask(
    __name__,
    static_folder=os.path.join(PROJECT_ROOT, "frontend"),
    static_url_path="",
)
CORS(app)

# ---------- 单例 Estimator ----------
_estimator: Optional[FoodCalorieEstimator] = None


def _latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """自动取目录下最新的 best.pth / epoch_*.pth / last.pth"""
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = []
    for pattern in ("best.pth", "last.pth"):
        p = os.path.join(checkpoint_dir, pattern)
        if os.path.exists(p):
            candidates.append((os.path.getmtime(p), p))
    for f in glob.glob(os.path.join(checkpoint_dir, "*.pth")):
        candidates.append((os.path.getmtime(f), f))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _resolve_ckpt(*names) -> Optional[str]:
    """按顺序在 checkpoints/* 目录里找可用的 checkpoint"""
    ckpt_root = os.path.join(PROJECT_ROOT, "checkpoints")
    for name in names:
        sub = os.path.join(ckpt_root, name)
        latest = _latest_checkpoint(sub)
        if latest:
            return latest
    return None


def get_estimator() -> FoodCalorieEstimator:
    global _estimator
    if _estimator is None:
        config_path = os.path.join(PROJECT_ROOT, "config.yaml")
        nir_ckpt = _resolve_ckpt("nir_generator")
        multitask_ckpt = _resolve_ckpt("multitask")
        baseline_ckpt = _resolve_ckpt("baseline")

        log.info("Loading estimator with checkpoints:")
        log.info("  nir_generator: %s", nir_ckpt or "(none)")
        log.info("  multitask:     %s", multitask_ckpt or "(none)")
        log.info("  baseline:      %s", baseline_ckpt or "(none)")

        _estimator = FoodCalorieEstimator(
            config_path=config_path,
            nir_ckpt=nir_ckpt,
            multitask_ckpt=multitask_ckpt,
            baseline_ckpt=baseline_ckpt,
        )
    return _estimator


# ---------- 路由 ----------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)


@app.route("/api/health", methods=["GET"])
def health_check():
    try:
        status = get_estimator().get_model_status()
        return jsonify({"status": "ok", "data": status})
    except Exception as e:
        log.exception("health check failed")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "no image file in form-data"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"success": False, "error": "empty filename"}), 400
    try:
        image_bytes = file.read()
        est = get_estimator()
        result = est.predict_all(image_bytes)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        log.exception("predict failed")
        return jsonify({"success": False, "error": str(e),
                        "trace": traceback.format_exc()}), 500


@app.route("/api/predict-nir", methods=["POST"])
def predict_nir():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "no image file"}), 400
    try:
        image_bytes = request.files["image"].read()
        est = get_estimator()
        nir_pil, _ = est.predict_nir(image_bytes)
        buf = io.BytesIO()
        nir_pil.save(buf, format="PNG")
        return jsonify({
            "success": True,
            "data": {"nir_image": base64.b64encode(buf.getvalue()).decode("utf-8")},
        })
    except Exception as e:
        log.exception("predict-nir failed")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict-calories", methods=["POST"])
def predict_calories():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "no image file"}), 400
    try:
        image_bytes = request.files["image"].read()
        use_nir = request.form.get("use_nir", "true").lower() == "true"
        est = get_estimator()
        result = est.predict_multitask(image_bytes, use_nir=use_nir)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        log.exception("predict-calories failed")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sample-images", methods=["GET"])
def sample_images():
    """返回几个内置示例图 (base64) 供前端快捷测试"""
    sample_dir = os.path.join(PROJECT_ROOT, "frontend", "samples")
    samples = []
    if os.path.isdir(sample_dir):
        for fn in sorted(os.listdir(sample_dir)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                with open(os.path.join(sample_dir, fn), "rb") as f:
                    samples.append({
                        "name": fn,
                        "image": base64.b64encode(f.read()).decode("utf-8"),
                    })
    return jsonify({"success": True, "data": {"samples": samples}})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info("Starting Food Calorie Estimation server on port %d", port)
    log.info("Frontend: http://localhost:%d/", port)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
