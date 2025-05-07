# inference.py
# ------------------------------------------------------------
# 功能：
#   1. 讀取配置與命令列參數
#   2. 載入 .mat 數據並推理 PINN 模型
#   3. 計算誤差指標並輸出至日誌 / JSON
#   4. 繪製預測圖像或僅展示
# ------------------------------------------------------------
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from DataPrep.data import load_data
from Allocation.logger import get_logger
from model import PINNInferer
from AnalyzeVisualize.visualisation import plot_prediction

logger = get_logger("PINN-Inference")

# ------------------------ 工具函式 --------------------------

def load_cfg(path: Path) -> Dict[str, Any]:
    """讀取 JSON 或 YAML 配置檔（僅示範 JSON）。"""
    with path.open(encoding="utf-8") as f:
        return json.load(f)

# ------------------------ 誤差計算 --------------------------

def evaluate_pinn(
    pinn: PINNInferer,
    X_star: np.ndarray,
    u_star: np.ndarray,
    true_params: Dict[str, float],
    device: torch.device,
) -> Dict[str, float]:
    """計算相對 L2 誤差與物理參數誤差，返回指標字典。"""
    try:
        X_tensor = torch.as_tensor(X_star, dtype=torch.float32, device=device)
        x_s, t_s = X_tensor[:, :1], X_tensor[:, 1:2]
        with torch.no_grad():
            u_pred = pinn(x_s, t_s).cpu().numpy()

        # 相對 L2
        err_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

        # 物理常數誤差（依 PDE 不同自動判斷）
        metrics = {"rel_L2_u": float(err_u)}
        if "lambda1" in true_params:
            pred = float(pinn.physics["lambda1"].item())
            metrics["lambda1_err_%"] = abs(pred - true_params["lambda1"]) / abs(true_params["lambda1"]) * 100
        if "lambda2" in true_params:
            pred = float(torch.exp(pinn.physics["lambda2"]).item())
            metrics["lambda2_err_%"] = abs(pred - true_params["lambda2"]) / abs(true_params["lambda2"]) * 100

        # 紀錄到日誌
        for k, v in metrics.items():
            logger.info(f"{k:14s}: {v:.4e}")
        return {**metrics, "u_pred": u_pred}  # 將預測值也一併返回
    except Exception:
        logger.exception("評估失敗！")
        raise

# ------------------------ 主程式封裝 ------------------------

def run_inference(cfg: Dict[str, Any], args: argparse.Namespace):
    """根據 cfg 與 CLI 參數執行推理流程。"""
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    # 1. 讀取資料
    logger.info("載入推理資料 …")
    X_star, u_star, lb, ub = load_data(args.file_path)

    # 2. 組建模型並載入權重
    logger.info("構建模型並載入檢查點 …")
    pinn = PINNInferer(cfg["layers"], lb, ub).to(device)
    pinn = PINNInferer.load(args.model_path, layers=cfg["layers"], lb=lb, ub=ub, device=device)

    # 3. 評估
    logger.info("開始評估 …")
    true_params = cfg.get("pde_params", {})
    results = evaluate_pinn(pinn, X_star, u_star, true_params, device)

    # 4. 繪圖或展示
    if not args.no_plot:
        logger.info("繪製預測圖像 …")
        plot_prediction(X_star, X_star[:, 1], results["u_pred"], save_path=args.save_path, show=args.show)

    # 5. 儲存指標
    if args.metrics_path:
        mp = Path(args.metrics_path)
        mp.parent.mkdir(parents=True, exist_ok=True)
        with mp.open("w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in results.items() if k != "u_pred"}, f, indent=2)
        logger.info("指標已寫入 %s", mp.resolve())

# ------------------------ CLI 參數 ---------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PINN推理模型")
    p.add_argument("--cfg", default="config.json", help="配置檔路徑")
    p.add_argument("--file_path", help=".mat 資料檔", required=False)
    p.add_argument("--model_path", help="模型檔 (.pth)")
    p.add_argument("--save_path", default="prediction.png", help="圖像輸出路徑")
    p.add_argument("--metrics_path", default="", help="指標輸出 JSON 路徑")
    p.add_argument("--device", default="", help="cpu / cuda:0 / mps")
    p.add_argument("--show", action="store_true", help="顯示圖像而不保存")
    p.add_argument("--no_plot", action="store_true", help="不產生圖像")
    return p.parse_args()

# ------------------------------ main -------------------------

def main():
    args = get_args()
    cfg = load_cfg(Path(args.cfg))

    # CLI 參數覆寫配置檔
    for k, v in vars(args).items():
        if v and k in cfg:
            cfg[k] = v

    try:
        run_inference(cfg, args)
    except Exception:
        logger.exception("推理流程失敗！")
        sys.exit(1)


if __name__ == "__main__":
    main()
