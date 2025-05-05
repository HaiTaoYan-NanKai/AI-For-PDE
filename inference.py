import numpy as np
import torch
import argparse
from config import CONFIG
from data import load_data
from model import PINNInferer
from visualize import plot_prediction
from logger import get_logger

logger = get_logger('Inference')

# ----------------------- 評估與測試 -------------------------
def evaluate_pinn(pinn, X_star, u_star, nu_true):
    """評估訓練好的 PINN 並計算誤差"""
    try:
        X_star_tensor = torch.tensor(X_star, dtype=torch.float32, device=DEVICE)
        x_s, t_s = X_star_tensor[:, 0:1], X_star_tensor[:, 1:2]
        with torch.no_grad():
            u_pred = pinn.net_u(x_s, t_s).cpu().numpy()
        err_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

        lambda1_val = pinn.lambda1.item()
        lambda2_val = torch.exp(pinn.raw_lambda2).item()
        err_l1 = abs(lambda1_val - 1.0) * 100
        err_l2 = abs(lambda2_val - nu_true) / nu_true * 100

        logger.info(f"相對 L2 誤差 u : {err_u:.2e}")
        logger.info(f"λ1 誤差 (%)    : {err_l1:.3f}")
        logger.info(f"λ2 誤差 (%)    : {err_l2:.3f}")
        return u_pred
    except Exception as e:
        logger.error(f"模型評估失敗：{str(e)}")
        raise

# ----------------------- 推理主程序 ----------------------
if __name__ == '__main__':
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='推理 PINN 模型')
    parser.add_argument('--file_path', type=str, default=CONFIG['file_path'], help='數據文件路徑')
    parser.add_argument('--model_path', type=str, default=CONFIG['model_path'], help='模型加載路徑')
    parser.add_argument('--save_path', type=str, default='prediction.png', help='預測圖像保存路徑')
    args = parser.parse_args()

    try:
        nu_true = 0.01 / np.pi  # λ2 的真實值

        # 從配置文件和命令行加載超參數
        layers = CONFIG['layers']
        file_path = args.file_path
        model_path = args.model_path
        save_path = args.save_path

        # 加載推理數據
        logger.info("開始加載推理數據")
        X_star, u_star, lb, ub = load_data(file_path)

        # 構建模型並加載參數
        logger.info("開始構建 PINNInferer")
        pinn = PINNInferer(layers, lb, ub)
        pinn.load_model(model_path)

        # 評估模型
        logger.info("開始評估模型")
        u_pred = evaluate_pinn(pinn, X_star, u_star, nu_true)

        # 繪製結果
        logger.info("開始繪製預測圖像")
        plot_prediction(X_star, X_star[:, 1], u_pred, save_path=save_path)
    except Exception as e:
        logger.error(f"推理主程序失敗：{str(e)}")
        raise