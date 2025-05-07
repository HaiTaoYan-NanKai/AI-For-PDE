# pinn_model.py
# ------------------------------------------------------------
# 功能：
#   1. 定義全連接神經網絡（FCNN）作為基礎網絡組件
#   2. 實現PINN基類，支持數據歸一化、前向傳播及模型保存/加載
#   3. 提供Burgers方程訓練器，計算PDE殘差並配置優化器
#   4. 提供推理專用類，簡化前向傳播過程
# ------------------------------------------------------------
from __future__ import annotations
import os
from pathlib import Path
from typing import Sequence, Callable, Optional, Dict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from Allocation.logger import get_logger
logger = get_logger("Model")

# -------------------------- 網絡組件 ------------------------ #
class FCNN(nn.Sequential):
    """
        初始化全連接神經網絡。
        Args:
            layers (Sequence[int]): 網絡各層的維度序列。
            act_cls (Callable[[], nn.Module], optional): 激活函數類，默認為 nn.Tanh。
            w_init (Callable[[Tensor], None], optional): 權重初始化函數，默認為 nn.init.xavier_uniform_。
    """
    def __init__(self,
                 layers:Sequence[int],
                 act_cls:Callable[[],nn.Module]=nn.Tanh,
                 w_init:Callable[[Tensor],None]=nn.init.xavier_uniform_,):
        modules = []
        for in_dim, out_dim in zip(layers[:-2],layers[1:-1]):
            linear = nn.Linear(in_dim, out_dim)
            w_init(linear.weight);nn.init.zeros_(linear.bias)
            modules += [linear, act_cls()]
        
        # 輸出
        linear = nn.Linear(layers[-2], layers[-1])
        w_init(linear.weight);nn.init.zeros_(linear.bias)
        modules.append(linear)
        super().__init__(*modules)


# -------------------------- 網絡容器基建 ------------------------ #
class PINNBase(nn.Module):
    def __init__(
            self,
            layers:Sequence[int],
            lb: Sequence[float],
            ub: Sequence[float],
            act_cls: Callable[[], nn.Module] = nn.Tanh,
            device: Optional[str|torch.device] = None,
    ):
        super().__init__()
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lb = torch.as_tensor(lb, dtype=torch.float32, device=self.device)
        self.ub = torch.as_tensor(ub, dtype=torch.float32, device=self.device)
        self.net = FCNN(layers, act_cls).to(self.device)

        # 物理參數
        self.physics:nn.ParameterDict = nn.ParameterDict(dict(
            lambda1 = nn.Parameter(torch.zeros(1, device=self.device)),
            lambda2 = nn.Parameter(torch.tensor([6.0], device=self.device))            
        ))

        logger.info(f"PINN初始化完成 | layers={layers} | device={self.device}")

    # ************************** 接口：子類實現 ************************ #
    def pde_residual(self, x: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError("pde_residual 方法未實現")
    
    # ************************** 內部工具 ************************ #
    def _normalize(self, x: Tensor, t: Tensor) -> Tensor:
        X = torch.cat([x, t], dim=-1)
        return 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
    
    # ************************** 前向 ************************ #
    def forward(self, x:Tensor, t:Tensor) -> Tensor:
        Xn = self._normalize(x, t)
        return self.net(Xn)
    
    # ************************** 保存參數 ************************ #
    def save(self, path:str|os.PathLike = "model.pth", meta:Optional[Dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = dict(
            net = self.net.state_dict(),
            physics = {k:p.detach().cpu() for k,p in self.physics.items()},
            meta = meta or {}
        )
        torch.save(ckpt, path)
        logger.info(f"模型已保存->{path.resolve()}")

    @classmethod
    def load(cls, path:str|os.PathLike, **init_kwargs) -> "PINNBase":
        ckpt = torch.load(path, map_location = model.device)
        model:"PINNBase" = cls(**init_kwargs)
        model.net.load_state_dict(ckpt["net"])
        for k,v in ckpt["physics"].items():
            model.physics[k].data.copy_(v.to(model.device))
        logger.info(f"模型已从{path}加載")
        return model


# -------------------------- 網絡容器：Burgers訓練器 ------------------------ #
class BurgersTrainer(PINNBase):
    def __init__(
            self,
            X_u:Tensor|np.ndarray,
            u:Tensor|np.ndarray,
            layers:Sequence[int],
            lb:Sequence[float],
            ub:Sequence[float],
            adam_lr:float = 1e-3,
            lbfgs_max_iter:int = 20000,
            **kwargs,
    ):
        super().__init__(layers, lb, ub, **kwargs)

        X_u = torch.as_tensor(X_u, dtype=torch.float32, device=self.device)
        u = torch.as_tensor(u, dtype=torch.float32, device=self.device)
        self.x = X_u[:,:1].requires_grad_(True)
        self.t = X_u[:,1:2].requires_grad_(True)
        self.u_obs = u

        # 優化器
        self.opt_adam = torch.optim.Adam(self.parameters(), lr = adam_lr)
        self.opt_lbfgs = torch.optim.LBFGS(
            self.parameters(),
            max_iter=lbfgs_max_iter,
            tolerance_grad=1e-9,
            tolerance_change=1e-9,
            history_size=50,
            line_search_fn="strong_wolfe",

        )

        logger.info("BurgersTrainer初始化完成")

    # ************************** PDE殘差 ************************ #
    def pde_residual(self, x: Tensor, t: Tensor) -> Tensor:
        # 調用 BurgersTrainer 類實例的 __call__ 方法
        # 最終調用forward方法
        u = self(x,t)
        u_grads = torch.autograd.grad(u, (x,t), torch.ones_like(u), create_graph = True, retain_graph=True)
        u_x, u_t = u_grads[0], u_grads[1]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]

        lambda1 = self.physics["lambda1"]
        lambda2 = self.physics["lambda2"]

        return lambda1 * u_t + lambda2 * u * u_x - u_xx


# -------------------------- 網絡容器：推理優化器 ------------------------ #
class PINNInferer(PINNBase):
    """仅推理：直接 forward 即可"""
    pass