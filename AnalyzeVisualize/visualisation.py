import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from Allocation.logger import get_logger

logger = get_logger('Visualize')

def plot_prediction(X, T, u_pred, save_path='prediction.png'):
    """繪製預測解的圖像"""
    try:
        U_pred = griddata(X, u_pred.flatten(), (X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1)), method='cubic')
        plt.figure(figsize=(5, 2))
        plt.imshow(U_pred.T, extent=[T.min(), T.max(), X[:, 0].min(), X[:, 0].max()], origin='lower', aspect='auto', cmap='rainbow')
        plt.colorbar()
        plt.xlabel('t')
        plt.ylabel('x')
        plt.title('預測的 u(t,x)')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"預測圖像已保存至 {save_path}")
    except Exception as e:
        logger.error(f"繪製圖像失敗：{str(e)}")
        raise