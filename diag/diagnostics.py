import json
import os
import time
from typing import Any

import torch
import torch.nn.functional as F


class DiagnosticLogger:
    def __init__(
        self,
        model,
        fisher_ema,
        out_dir: str,
        run_id: str,
        user_config: dict[str, Any],
        topk_ratio: float = 0.1,
    ):
        self.model = model
        self.fisher_ema = fisher_ema
        self.out_dir = out_dir
        self.run_id = run_id
        self.user_config = user_config
        self.topk_ratio = topk_ratio
        self.eps = 1e-12

        self.params = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        self.anchors = {
            name: p.detach().float().cpu().clone()
            for name, p in self.params
        }
        self.metrics_path = os.path.join(self.out_dir, "metrics.jsonl")
        os.makedirs(self.out_dir, exist_ok=True)
        self._write_metadata()

    def _write_metadata(self):
        metadata = {
            "run_id": self.run_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "topk_ratio": self.topk_ratio,
            "anchor": "outer_step_0_fp32_cpu",
            "user_config": self.user_config,
        }
        with open(os.path.join(self.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    @torch.no_grad()
    def update_fisher_from_optimizers(self, optimizers):
        if self.fisher_ema is None:
            return
        seen = set()
        params = []
        for opt in optimizers:
            for group in opt.param_groups:
                for p in group["params"]:
                    key = id(p)
                    if key in seen:
                        continue
                    seen.add(key)
                    params.append(p)
        self.fisher_ema.update(params)

    @staticmethod
    def _to_float(value):
        return None if value is None else float(value)

    def _safe_cosine(self, a: torch.Tensor, b: torch.Tensor):
        if a.numel() == 0 or b.numel() == 0:
            return None
        a = a.flatten()
        b = b.flatten()
        denom = a.norm() * b.norm()
        if denom.item() <= self.eps:
            return None
        return F.cosine_similarity(a, b, dim=0, eps=self.eps).item()

    @torch.no_grad()
    def compute_s1(self):
        if self.fisher_ema is None:
            return {
                "s1_global": None,
                "s1_rel_global": None,
                "s1_param_count": 0,
                "s1_topk_count": 0,
            }

        weighted_cos_drift = 0.0
        weighted_rel_drift = 0.0
        total_weight = 0
        param_count = 0

        for name, p in self.params:
            if not self.fisher_ema.has(p):
                continue
            anchor = self.anchors.get(name)
            if anchor is None or anchor.shape != p.shape:
                continue

            fisher = self.fisher_ema.F[id(p)].detach().float().cpu().flatten()
            if fisher.numel() == 0:
                continue
            k = max(1, int(fisher.numel() * self.topk_ratio))
            k = min(k, fisher.numel())
            _, idx = torch.topk(fisher, k=k, largest=True, sorted=False)

            current = p.detach().float().cpu().flatten()
            anchor_flat = anchor.flatten()
            delta = current - anchor_flat
            delta_topk = delta[idx]
            anchor_topk = anchor_flat[idx]

            cos = self._safe_cosine(delta_topk, anchor_topk)
            rel = delta_topk.norm().item() / (anchor_topk.norm().item() + self.eps)
            if cos is None:
                continue

            weighted_cos_drift += (1.0 - cos) * k
            weighted_rel_drift += rel * k
            total_weight += k
            param_count += 1

        if total_weight == 0:
            return {
                "s1_global": None,
                "s1_rel_global": None,
                "s1_param_count": 0,
                "s1_topk_count": 0,
            }
        return {
            "s1_global": weighted_cos_drift / total_weight,
            "s1_rel_global": weighted_rel_drift / total_weight,
            "s1_param_count": param_count,
            "s1_topk_count": total_weight,
        }

    @torch.no_grad()
    def compute_s2(self):
        components = {}
        params = dict(self.params)
        embedding_anchor = self.anchors.get("transformer.wte.weight")
        embedding_rms = None
        if embedding_anchor is not None and embedding_anchor.numel() > 0:
            embedding_rms = embedding_anchor.norm().item() / (embedding_anchor.numel() ** 0.5)

        for key, name in (
            ("embedding", "transformer.wte.weight"),
            ("lm_head", "lm_head.weight"),
        ):
            param = params.get(name)
            anchor = self.anchors.get(name)
            if param is None or anchor is None:
                continue
            current = param.detach().float().cpu()
            delta = current - anchor
            delta_norm = delta.norm().item()
            anchor_norm = anchor.norm().item()
            rms_drift = delta_norm / (delta.numel() ** 0.5)

            if anchor_norm > self.eps:
                rel = delta_norm / anchor_norm
                cos = self._safe_cosine(current, anchor)
                cos_drift = None if cos is None else 1.0 - cos
                score = rel if cos_drift is None else 0.5 * (rel + cos_drift)
                norm_mode = "relative_to_anchor"
            else:
                rel = None
                cos_drift = None
                normalizer = embedding_rms if embedding_rms is not None and embedding_rms > self.eps else 1.0
                score = rms_drift / normalizer
                norm_mode = "rms_drift_over_embedding_rms"

            components[key] = {
                "rel_norm": rel,
                "anchor_norm": anchor_norm,
                "rms_drift": rms_drift,
                "cosine_drift": cos_drift,
                "norm_mode": norm_mode,
                "score": score,
            }

        scores = [item["score"] for item in components.values()]
        s2_global = sum(scores) / len(scores) if scores else None
        return {
            "s2_global": s2_global,
            "s2_embedding": components.get("embedding", {}),
            "s2_lm_head": components.get("lm_head", {}),
        }

    @torch.no_grad()
    def write(
        self,
        *,
        step: int,
        outer_step_idx: int,
        is_outer_step: bool,
        train_loss=None,
        val_bpb=None,
        core_metric=None,
        learning_rate_multiplier=None,
        total_training_time=None,
        record_type: str = "outer_step",
    ):
        s1 = self.compute_s1()
        s2 = self.compute_s2()
        s1_global = s1["s1_global"]
        s2_global = s2["s2_global"]
        jds_raw = None if s1_global is None or s2_global is None else s1_global + s2_global

        record = {
            "run_id": self.run_id,
            "record_type": record_type,
            "timestamp": time.time(),
            "step": int(step),
            "outer_step_idx": int(outer_step_idx),
            "is_outer_step": bool(is_outer_step),
            "train_loss": self._to_float(train_loss),
            "val_bpb": self._to_float(val_bpb),
            "core_metric": self._to_float(core_metric),
            "learning_rate_multiplier": self._to_float(learning_rate_multiplier),
            "total_training_time": self._to_float(total_training_time),
            "s1_global": self._to_float(s1_global),
            "s1_rel_global": self._to_float(s1["s1_rel_global"]),
            "s1_param_count": int(s1["s1_param_count"]),
            "s1_topk_count": int(s1["s1_topk_count"]),
            "s2_global": self._to_float(s2_global),
            "s2_embedding": s2["s2_embedding"],
            "s2_lm_head": s2["s2_lm_head"],
            "jds_raw": self._to_float(jds_raw),
        }
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        return record
