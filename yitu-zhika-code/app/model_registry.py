"""模型版本注册表。统一管理多个已审计模型，支持版本切换。

每个版本对应一组已审计的 checkpoint + 审计文件 + 协议文件。
切换版本时检查 SHA256，确保权重未被篡改。
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ModelEntry:
    """一个已审计的模型版本。"""
    name: str                # 版本名（如 meal_nir_official_v1）
    checkpoint: Path         # best.pt 路径
    test_metrics: dict       # 测试指标
    protocol: dict           # 训练协议
    audit: dict              # 审计结果
    checkpoint_sha256: str   # 权重 SHA256


class ModelRegistry:
    """模型版本注册表。"""

    def __init__(self):
        self._entries: dict[str, ModelEntry] = {}
        self._load_all()

    def _load_all(self):
        """启动时扫描所有已审计的模型。"""
        results_dir = ROOT / "results"
        if not results_dir.exists():
            return
        for d in results_dir.iterdir():
            if not d.is_dir():
                continue
            entry = self._try_load(d.name)
            if entry:
                self._entries[d.name] = entry

    def _try_load(self, name: str) -> Optional[ModelEntry]:
        """尝试加载一个模型版本。失败返回 None。"""
        folder = ROOT / "results" / name
        ckpt = ROOT / "checkpoints" / name / "best.pt"
        metrics_f = folder / "test_metrics.json"
        protocol_f = folder / "protocol.json"
        audit_files = [
            folder / "completion_audit.json",
            folder / "paired_completion_audit.json",
        ]

        if not ckpt.exists() or not metrics_f.exists():
            return None

        try:
            metrics = json.loads(metrics_f.read_text(encoding="utf-8"))
            protocol = json.loads(protocol_f.read_text(encoding="utf-8")) if protocol_f.exists() else {}
            audit = None
            for af in audit_files:
                if af.exists():
                    audit = json.loads(af.read_text(encoding="utf-8"))
                    break
            if audit is None:
                return None

            # 计算 SHA256
            from scripts.capsicum_job import file_digest
            sha = file_digest(ckpt)

            return ModelEntry(
                name=name,
                checkpoint=ckpt,
                test_metrics=metrics,
                protocol=protocol,
                audit=audit,
                checkpoint_sha256=sha,
            )
        except Exception:
            return None

    def list_versions(self) -> list[str]:
        """列出所有已审计的模型版本。"""
        return list(self._entries.keys())

    def get(self, name: str) -> Optional[ModelEntry]:
        """获取指定版本。"""
        return self._entries.get(name)

    def verify(self, name: str) -> bool:
        """验证模型权重 SHA256 是否与审计一致。"""
        entry = self._entries.get(name)
        if not entry:
            return False
        return (entry.checkpoint_sha256
                == entry.test_metrics.get("checkpoint_sha256")
                == entry.audit.get("checkpoint_sha256"))

    def info(self, name: str) -> dict:
        """获取模型信息。"""
        entry = self._entries.get(name)
        if not entry:
            return {"error": f"model not found: {name}"}
        return {
            "name": entry.name,
            "checkpoint": str(entry.checkpoint),
            "checkpoint_sha256": entry.checkpoint_sha256,
            "verified": self.verify(name),
            "test_metrics": entry.test_metrics.get("test", {}),
            "best_epoch": entry.test_metrics.get("best_epoch"),
            "manifest_sha256": entry.test_metrics.get("manifest_sha256"),
        }