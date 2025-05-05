import numpy as np
import scipy.io
from logger import get_logger

logger = get_logger('Data')

def load_data(file_path, N_u=None, noise_level=0.0):
    """從 .mat 文件加載數據，支持訓練和推理"""
    try:
        data = scipy.io.loadmat(file_path)
        t = data.get('t', None)
        x = data.get('x', None)
        usol = data.get('usol', None)
        if t is None or x is None or usol is None:
            raise KeyError("數據文件中缺少 't', 'x' 或 'usol' 字段")
        
        t = t.flatten()[:, None]  # 時間數據
        x = x.flatten()[:, None]  # 空間數據
        Exact = np.real(usol).T  # 真實解

        X, T = np.meshgrid(x, t)
        X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        u_star = Exact.flatten()[:, None]

        lb, ub = X_star.min(0), X_star.max(0)  # 數據的上下界
        logger.info(f"數據加載成功：{file_path}, lb={lb}, ub={ub}")

        if N_u is not None:
            # 訓練模式：選擇訓練點並可添加噪聲
            if N_u > X_star.shape[0]:
                raise ValueError(f"N_u ({N_u}) 超過數據點數 ({X_star.shape[0]})")
            idx = np.random.choice(X_star.shape[0], N_u, replace=False)
            X_u_train = X_star[idx, :]
            u_train = u_star[idx, :]
            if noise_level > 0:
                u_train += noise_level * np.std(u_train) * np.random.randn(*u_train.shape)
                logger.info(f"已添加噪聲，noise_level={noise_level}")
            return X_u_train, u_train, X_star, u_star, lb, ub
        else:
            # 推理模式：僅返回必要數據
            return X_star, u_star, lb, ub
    except FileNotFoundError:
        logger.error(f"數據文件未找到：{file_path}")
        raise
    except Exception as e:
        logger.error(f"數據加載失敗：{str(e)}")
        raise