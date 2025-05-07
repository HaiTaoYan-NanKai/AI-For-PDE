# pinn_loss.py
# ------------------------------------------------------------
# 功能：
#   1. 定義支持的損失項（MSE、L1、L2正則等）並可擴展
#   2. 提供損失權重容器，支持動態調整
#   3. 實現PINN損失計算，包含物理與數學損失項
#   4. 支持損失項監控（可選功能，已註釋）
# ------------------------------------------------------------
from __future__ import annotations
import enum
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping
import torch
import torch.nn.functional as F

# ---------------------- 日誌工具 ------------------------ #
from Allocation.logger import get_logger
logger = get_logger("PINN-Loss")

# ---------------------- 定義所有支持的損失項 ------------------------ #
class Term(enum.Enum):
    """
    如新增損失項, 請在這裡添加, 並在'_compute_components'中實現即可
    """
    # 物理損失項
    MSE_U = enum.auto()
    # 數學損失項
    MSE_F = enum.auto()
    # 解u的L_1誤差
    L1_U = enum.auto()
    # PDE殘差的L_1誤差
    L1_F = enum.auto()
    # 參數L_2正則
    L2_REG = enum.auto()

# ---------------------- 權重容器 ------------------------ #
@dataclass
class LossWeights:
    values:Dict[Term,float]=field(
        default_factory = lambda:{
            Term.MSE_U:1.0,
            Term.MSE_F:1.0,
            Term.L1_U:0.0,
            Term.L1_F:0.0,
            Term.L2_REG:0.0,
        }
    )

    def __getitem__(self, term:Term) -> float:
        """
        根據損失項獲取對應的權重值。
        Args:term (Term): 損失項。
        Returns:float: 對應損失項的權重值，若不存在則返回 0.0。
        """
        return self.values.get(term,0.0)
    
    def __repr__(self) -> str:
         return ", ".join(f"{t.name}={w}" for t, w in self.values.items())
    
# ---------------------- 損失容器 ------------------------ #
class PINNLoss:
    """
    PINN 复合损失计算器。
    参数
    ------
    pinn : object
        必须包含以下属性 / 方法：
        ``x``、``t``、``u_obs``、``net_u``、``net_f``、``net``。
    weights : LossWeights, optional
        各项损失权重；若为 None 则使用默认值。
    reduction : Callable, optional
        张量到标量的归约函数，默认 `torch.mean`。
    提示
    ----
    - 若使用优化器自带的 ``weight_decay`` 处理 L2 正则，则将 `L2_REG` 权重设为 0 即可。
    - 如需获得各分项数值，可调用 `components()`（示例代码已注释）。
    """
    # IDE類型提示：pinn需具備的屬性名稱
    required_pinn_attrs = ("x","t","u_obs","net_u","net_f","net")
    def __init__(
            self,
            pinn,
            *,
            weights:LossWeights|None = None,
            # 歸約函數，將多維張量轉換為單維
            reduction:Callable[[torch.Tensor],torch.Tensor] = torch.mean,
    ) -> None:
        # =========== 檢查pinn是否具備必要屬性 =========== #
        missing = [attr for attr in self.required_pinn_attrs if not hasattr(pinn, attr)]
        if missing:
            raise ValueError(f"pinn缺少必要屬性：{missing}")
        
        self.pinn = pinn
        self.weights = weights or LossWeights()
        self.reduction = reduction
        logger.info("損失函數初始化，權重配置：%s",self.weights)
    
    # *************************** 公共接口 ************************ #
    def __call__(self) -> torch.Tensor:
        """計算并返回加權后的總損失"""
        comps = self._compute_components()
        total = torch.zeros(1,device=self.pinn.x.device)

        # 逐漸累加
        for term, value in comps.items():
            w = self.weights[term]
            if w >0:
                contrib = w * value
                total += contrib
                logger.debug("%s:%.3e(w=%3f)", term.name.ljust(), value.item(),w)
        
        logger.debug("總算是：%.3e", total.item())
        return total.squeeze()
    
    # *************************** 私有方法 ************************ #
    def _compute_components(self) -> Mapping[Term, torch.Tensor]:
        """計算各損失分量，返回映射"""
        x = self.pinn.x
        t = self.pinn.t
        u_obs = self.pinn.u_obs

        # 單詞前向，避免重複計算
        u_pred = self.pinn.net_u(x,t)
        f_pred = self.pinn.net_f(x,t)

        # 計算各項
        mse_u = F.mse_loss(u_pred,u_obs,reduction="mean")
        mse_f = self.reduction(f_pred**2)
        l1_u = F.l1_loss(u_pred,u_obs,reduction="mean")
        l1_f = self.reduction(f_pred.abs())
        params = torch.norm(torch.nn.utils.parameters_to_vector(self.pinn.net.parameters().to(self.pinn.x.device))**2)
        l2_reg = torch.norm(params,p=2)**2

        return {
            Term.MSE_U:mse_u,
            Term.MSE_F:mse_f,
            Term.L1_U:l1_u,
            Term.L1_F:l1_f,
            Term.L2_REG:l2_reg,
        }
    
    # ------------------------------------------------------------------
    # 可选功能 – 如需返回分项监控，可取消注释
    # ------------------------------------------------------------------
    #
    # def components(self) -> Dict[str, float]:
    #     """返回各损失分项（已 detach，CPU float）。"""
    #     comps = self._compute_components()
    #     return {term.name: value.detach().cpu().item() for term, value in comps.items()}

