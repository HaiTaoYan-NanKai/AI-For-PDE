import time
import numpy as np
import torch
import argparse
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from config import CONFIG
from data import load_data
from model import PINNTrainer
from loss import PINNLoss
from logger import get_logger

logger = get_logger('Train')

# ----------------------- 訓練循環 ----------------------------------
def train_pinn(pinn, loss_fn, n_adam=10000, scheduler_type='StepLR', step_size=2000, gamma=0.5, patience=1000, factor=0.5):
    """使用 Adam 和 LBFGS 優化器訓練 PINN，包含學習率調度"""
    try:
        t0 = time.time()
        
        # 初始化學習率調度器
        if scheduler_type == 'StepLR':
            scheduler = StepLR(pinn.opt_adam, step_size=step_size, gamma=gamma)
            logger.info(f"使用 StepLR 調度器，step_size={step_size}, gamma={gamma}")
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(pinn.opt_adam, mode='min', patience=patience, factor=factor)
            logger.info(f"使用 ReduceLROnPlateau 調度器，patience={patience}, factor={factor}")
        else:
            raise ValueError(f"不支持的調度器類型：{scheduler_type}")

        # Adam 優化
        for it in range(n_adam):
            pinn.opt_adam.zero_grad()
            L = loss_fn.compute()
            L.backward()
            pinn.opt_adam.step()

            # 更新學習率
            if scheduler_type == 'StepLR':
                scheduler.step()
            elif scheduler_type == 'ReduceLROnPlateau':
                scheduler.step(L)

            if it % 1000 == 0:
                current_lr = pinn.opt_adam.param_groups[0]['lr']
                logger.info(f"Adam {it:6d} | 損失 {L.item():.3e} | λ1 {pinn.lambda1.item():.4f} | λ2 {torch.exp(pinn.raw_lambda2).item():.6f} | 學習率 {current_lr:.3e}")
        
        logger.info("切換到 LBFGS …")

        # LBFGS 優化
        def closure():
            pinn.opt_lbfgs.zero_grad()
            L = loss_fn.compute()
            L.backward()
            return L
        pinn.opt_lbfgs.step(closure)
        final_loss = loss_fn.compute().item()
        logger.info(f"LBFGS 完成 | 最終損失 {final_loss:.3e}")
    except Exception as e:
        logger.error(f"訓練失敗：{str(e)}")
        raise

# ----------------------- 訓練主程序 ----------------------
if __name__ == '__main__':
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='訓練 PINN 模型')
    parser.add_argument('--N_u', type=int, default=CONFIG['N_u'], help='訓練點數量')
    parser.add_argument('--n_adam', type=int, default=CONFIG['n_adam'], help='Adam 優化迭代次數')
    parser.add_argument('--file_path', type=str, default=CONFIG['file_path'], help='數據文件路徑')
    parser.add_argument('--model_path', type=str, default=CONFIG['model_path'], help='模型保存路徑')
    parser.add_argument('--scheduler_type', type=str, default=CONFIG['scheduler']['type'], help='學習率調度器類型 (StepLR 或 ReduceLROnPlateau)')
    parser.add_argument('--step_size', type=int, default=CONFIG['scheduler']['step_size'], help='StepLR 步長')
    parser.add_argument('--gamma', type=float, default=CONFIG['scheduler']['gamma'], help='StepLR 衰減因子')
    parser.add_argument('--patience', type=int, default=CONFIG['scheduler']['patience'], help='ReduceLROnPlateau 耐心值')
    parser.add_argument('--factor', type=float, default=CONFIG['scheduler']['factor'], help='ReduceLROnPlateau 衰減因子')
    args = parser.parse_args()

    try:
        # 從配置文件和命令行加載超參數
        N_u = args.N_u
        layers = CONFIG['layers']
        file_path = args.file_path
        model_path = args.model_path
        n_adam = args.n_adam
        loss_weights = CONFIG['loss_weights']
        scheduler_type = args.scheduler_type
        step_size = args.step_size
        gamma = args.gamma
        patience = args.patience
        factor = args.factor

        # 加載數據
        logger.info("開始加載訓練數據")
        X_u_train, u_train, X_star, u_star, lb, ub = load_data(file_path, N_u=N_u, noise_level=0.0)

        # 構建模型
        logger.info("開始構建 PINNTrainer")
        pinn = PINNTrainer(X_u_train, u_train, layers, lb, ub)

        # 初始化損失函數
        loss_fn = PINNLoss(pinn, weights=loss_weights)

        # 訓練模型
        logger.info("開始訓練模型")
        train_pinn(pinn, loss_fn, n_adam=n_adam, scheduler_type=scheduler_type, 
                   step_size=step_size, gamma=gamma, patience=patience, factor=factor)

        # 保存模型
        pinn.save_model(model_path)
    except Exception as e:
        logger.error(f"訓練主程序失敗：{str(e)}")
        raise