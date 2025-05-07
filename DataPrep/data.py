# data.py
# ------------------------------------------------------------
# 功能：
#   1. 從 .mat 文件加載數據並檢查必要字段
#   2. 預處理數據，生成網格並計算邊界
#   3. 篩選訓練點並可選添加噪聲
#   4. 提供訓練和推理模式的數據加載接口
# ------------------------------------------------------------
import numpy as np
import scipy.io as sio
from typing import Optional,Tuple
from Allocation.logger import get_logger

logger = get_logger("data")

#******************************** 文件加載 ***********************************#
def load_file(file_path:str) -> dict:
    try:
        data = sio.loadmat(file_path)
        required_keys = ['t','x','usol']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"數據文件缺少'{key}'字段")
        return data
    except FileNotFoundError:
        logger.error(f"數據文件未找到：{file_path}")
        raise
    except Exception as e:
        logger.error(f"數據記載失敗：{e}")
        raise

#******************************** 數據預處理 ***********************************#
def process_data(data:dict) ->  Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 時間數據
    t = data['t'].flatten()[:,None]
    # 空間數據
    x = data['x'].flatten()[:,None]
    # 真實解
    exact_usol = np.real(data['usol']).T

    X,T = np.meshgrid(x,t)
    # flatten() 將多維數組展平為一維數組
    # None 相當於 np.newaxis，用於在指定位置插入一個新的軸
    X_star = np.hstack((X.flatten()[:,None],T.flatten()[:,None]))
    u_star = exact_usol.flatten()[:,None]

    lb = X_star.min(0)
    ub = X_star.max(0)

    return X_star,u_star,lb,ub,x,t

#******************************** 數據篩選 ***********************************#
def select_training_points(X_star: np.ndarray, u_star: np.ndarray, N_u: int, noise_level: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    選擇訓練點並可添加噪聲。

    Args:
        X_star (np.ndarray): 所有數據點。
        u_star (np.ndarray): 所有真實解。
        N_u (int): 訓練點的數量。
        noise_level (float): 噪聲水平。

    Returns:
        Tuple[np.ndarray, np.ndarray]: X_u_train, u_train
    """
    if N_u > X_star.shape[0]:
        raise ValueError(f"N_u ({N_u}) 超過數據點數 ({X_star.shape[0]})")
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx, :]
    u_train = u_star[idx, :]
    if noise_level > 0:
        u_train += noise_level * np.std(u_train) * np.random.randn(*u_train.shape)
        logger.info(f"已添加噪聲, noise_level={noise_level}")
    return X_u_train, u_train

#******************************** 數據加載 ***********************************#
def load_data(file_path: str, N_u: Optional[int] = None, noise_level: float = 0.0) -> Tuple[np.ndarray, ...]:
    """
    從 .mat 文件加載數據，支持訓練和推理。

    Args:
        file_path (str): .mat 文件的路徑。
        N_u (Optional[int]): 訓練點的數量，若為 None 則為推理模式。默認為 None。
        noise_level (float): 噪聲水平。默認為 0.0。

    Returns:
        Tuple[np.ndarray, ...]: 
            訓練模式：X_u_train, u_train, X_star, u_star, lb, ub
            推理模式：X_star, u_star, lb, ub
    """
    data = load_file(file_path)
    X_star, u_star, lb, ub, _, _ = process_data(data)
    logger.info(f"數據加載成功：{file_path}, lb={lb}, ub={ub}")

    if N_u is not None:
        X_u_train, u_train = select_training_points(X_star, u_star, N_u, noise_level)
        return X_u_train, u_train, X_star, u_star, lb, ub
    else:
        return X_star, u_star, lb, ub