import torch
import torch.nn as nn
import os
from logger import get_logger

logger = get_logger('Model')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FCNN(nn.Module):
    """簡單的多層感知機：輸入 2D (x,t)，輸出 1D u"""
    def __init__(self, layers):
        super().__init__()
        modules = []
        for i in range(len(layers)-2):
            modules += [nn.Linear(layers[i], layers[i+1]), nn.Tanh()]
        modules += [nn.Linear(layers[-2], layers[-1])]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

class PINNBase:
    """PINN 基類，包含共享邏輯"""
    def __init__(self, layers, lb, ub):
        self.lb = torch.tensor(lb, dtype=torch.float32, device=DEVICE)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=DEVICE)
        self.net = FCNN(layers).to(DEVICE)
        self.lambda1 = nn.Parameter(torch.zeros(1, device=DEVICE))
        self.raw_lambda2 = nn.Parameter(torch.tensor([-6.0], device=DEVICE))
        logger.info(f"PINN 模型初始化，層結構：{layers}")

    def parameters(self):
        return list(self.net.parameters()) + [self.lambda1, self.raw_lambda2]

    def _normalize(self, x, t):
        X = torch.cat((x, t), dim=1)
        return 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

    def net_u(self, x, t):
        X_norm = self._normalize(x, t)
        return self.net(X_norm)

    def save_model(self, path='model.pth'):
        """保存模型參數"""
        try:
            torch.save({
                'net_state_dict': self.net.state_dict(),
                'lambda1': self.lambda1,
                'raw_lambda2': self.raw_lambda2
            }, path)
            logger.info(f"模型已保存至 {path}")
        except Exception as e:
            logger.error(f"模型保存失敗：{str(e)}")
            raise

    def load_model(self, path='model.pth'):
        """加載模型參數"""
        try:
            if os.path.exists(path):
                checkpoint = torch.load(path)
                self.net.load_state_dict(checkpoint['net_state_dict'])
                self.lambda1.data = checkpoint['lambda1']
                self.raw_lambda2.data = checkpoint['raw_lambda2']
                logger.info(f"模型已從 {path} 加載")
            else:
                raise FileNotFoundError(f"在 {path} 未找到模型")
        except Exception as e:
            logger.error(f"模型加載失敗：{str(e)}")
            raise

class PINNTrainer(PINNBase):
    """訓練專用的 PINN 類"""
    def __init__(self, X_u, u, layers, lb, ub):
        super().__init__(layers, lb, ub)
        try:
            self.x = torch.tensor(X_u[:, 0:1], dtype=torch.float32, device=DEVICE, requires_grad=True)
            self.t = torch.tensor(X_u[:, 1:2], dtype=torch.float32, device=DEVICE, requires_grad=True)
            self.u_obs = torch.tensor(u, dtype=torch.float32, device=DEVICE)
            self.opt_adam = torch.optim.Adam(self.parameters(), lr=1e-3)
            self.opt_lbfgs = torch.optim.LBFGS(self.parameters(),
                                               max_iter=50000, tolerance_grad=1e-9,
                                               tolerance_change=1e-9, history_size=50)
            logger.info("PINNTrainer 初始化完成")
        except Exception as e:
            logger.error(f"PINNTrainer 初始化失敗：{str(e)}")
            raise

    def net_f(self, x, t):
        try:
            lambda1 = self.lambda1
            lambda2 = torch.exp(self.raw_lambda2)
            u = self.net_u(x, t)
            u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            return u_t + lambda1 * u * u_x - lambda2 * u_xx
        except Exception as e:
            logger.error(f"PDE 殘差計算失敗：{str(e)}")
            raise

class PINNInferer(PINNBase):
    """推理專用的 PINN 類"""
    def __init__(self, layers, lb, ub):
        super().__init__(layers, lb, ub)
        logger.info("PINNInferer 初始化完成")