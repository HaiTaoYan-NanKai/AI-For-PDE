from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict,Optional
import numpy as np

import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import Summarywriter
from tqdm import tqdm

from DataPrep.data import load_data
from loss import PINNLoss
from Allocation.logger import get_logger
from model import BurgersTrainer

logger = get_logger("PINN-Train")

def set_seed(seed:int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_cfg(cfg_path:Path) -> Dict:
    with cfg_path.open(encoding="utf-8") as f:
        return json.load(f)

class TrainerWrapper:
    def __init__(self, cfg: Dict, args:argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.device = args.device

        X_u, u, X_all, u_all, lb, ub = load_data(
            args.filepath,
            N_u=args.N_u,
            noise_level=cfg.get("noise_level",0.0),
        )

        self.model = BurgersTrainer(X_u, u, cfg["layers"], lb, ub, device=self.device)

        # =========== 恢復模型 =========== #
        if args.resume and Path(args.resume).is_file():
            self.model = BurgersTrainer.load(
                args.resume,
                X_u =X_u,
                u=u,
                layers = cfg["layers"],
                lb = lb,
                ub = ub,
                device = self.device,
            )
            logger.info("Resume from checkpoint: %s", args.resume)
        
        # =========== 損失函數 =========== #
        self.loss_fn = PINNLoss(self.model, weights = cfg["loss_weights"])

        # =========== 學習率調度器 =========== #
        if args.scheduler == "StepLR":
            self.scheduler = StepLR(self.model.opt_adam,
                                    step_size=args.step_size,
                                    gamma=args.gamma)
        else:
            self.scheduler = ReduceLROnPlateau(self.model.opt_adam,
                                               mode="min",
                                               patience=args.patience,
                                               factor=args.factor)
        
        # =========== 監視器 =========== #
        self.writer:Optional[Summarywriter] = None
        if args.tb_dir:
            self.writer = Summarywriter(args.tb_dir)
            logger.info("Tensorboard writer initialized at %s", args.tb_dir)
        
        # =========== 保存目錄 =========== #
        self.ckpt_dir = Path(args.ckpt_dir) 
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # =========== 及時停止 =========== #
        self.early_patience = args.early_patience
        self.best_loss = float("inf")
        self.no_improve = 0

        # =========== 訓練參數 =========== #
        self.grad_clip = args.grad_clip
        self.log_interval = args.log_interval
        self.ckpt_freq = args.ckpt_freq
        self.n_adam = args.n_adam

    def clip_grad(self):
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
    
    def save_ckpt(self, name:str):
        path = self.ckpt_dir /name
        self.model.save(path, meta={"time":time.time()})
        logger.info("Checkpoint saved at %s", path)

    def train(self):
        try:
            # =========== Adam 階段 =========== #
            pbar = tqdm(range(self.n_adam), desc ="Adam", ncols = 90)
            for step in pbar:
                self.model.opt_adam.zero_grad()
                loss = self.loss_fn.compute()
                loss.backward()
                self.clip_grad()
                self.model.opt_adam.step()

                # =========== 更新調度器 =========== #
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(loss)
                else:
                    self.scheduler.step()
                
                if self.writer:
                    self.writer.add_scalar("loss/train", loss.item(), step)
                    self.writer.add_scalar("lr", self.model.opt_adam.param_groups[0]["lr"],step)
                
                if step % self.log_interval == 0:
                    lr = self.model.opt_adam.param_groups[0]["lr"]
                    logger.info("Adam %6d|loss %.3e|lr %.2e",step, loss.item(),lr)
                
                if step % self.ckpt_freq == 0 and step > 0:
                    self.save_ckpt(f"adam_{step}.pth")
                
                if loss.item() < self.best_loss - 1e-8:
                    self.best_loss = loss.item()
                    self.no_improve = 0
                else:
                    self.no_improve += 1
                    if self.no_improve >= self.early_patience:
                        logger.info("Early stopping triggered at step %d|best %.3e", step, self.best_loss)
                        break
            
                pbar.set_postfix(loss=f"{loss:.1e}")
        
            # =========== LBFGS 階段 =========== #
            logger.info("切換到優化器LBFGs")

            def closure():
                self.model.opt_lbfgs.zero_grad()
                l = self.loss_fn.compute()
                l.backward()
                return l
            self.model.opt_lbfgs.step(closure)
            final_loss = self.loss_fn.compute().item()
            logger.info("LBFGS finished | final loss %.3e",final_loss)
            self.save_ckpt("final.pth")
            if self.writer:
                self.writer.add_scalar("loss/final", final_loss)
        except KeyboardInterrupt:
            logger.warning("Interrupted by user.Saving checkpoint")
            self.save_ckpt("interrupted.pth")
        except Exception as exc:
            logger.exception("Trainning failed due to unexpected error")
            self.save_ckpt("error.pth")
            raise exc
        finally:
            if self.writer:
                self.writer.close()

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PINN model")
    p.add_argument("--cfg", default="config.json", help="config file path")
    p.add_argument("--file_path", help=".mat data file", required=False)
    p.add_argument("--N_u", type=int)
    p.add_argument("--n_adam", type=int)
    p.add_argument("--device", default="")
    # scheduler
    p.add_argument("--scheduler_type", choices=["StepLR", "ReduceLROnPlateau"])
    p.add_argument("--step_size", type=int)
    p.add_argument("--gamma", type=float)
    p.add_argument("--patience", type=int)
    p.add_argument("--factor", type=float)
    # monitoring / ckpt
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--tb_dir", default="runs")
    p.add_argument("--resume", help="checkpoint to resume from")
    p.add_argument("--log_interval", type=int, default=500)
    p.add_argument("--ckpt_freq", type=int, default=5000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--early_patience", type=int, default=3000)
    # misc
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = get_args()
    cfg = load_cfg(Path(args.cfg))

    # CLI ovverides config
    for k,v in vars(args).items():
        if v is not None and k in cfg:
            cfg[k] = v
    
    set_seed(args.seed)
    trainer = TrainerWrapper(cfg, args)
    trainer.train()
    
if __name__ == "__main__":
    main()

                


    