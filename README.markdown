# PINN 代碼架構模板

這是一個通用的物理信息神經網絡（PINN）代碼架構模板，基於 PyTorch 實現，適用於求解偏微分方程（PDE）等科學計算問題。模板設計簡潔高效、邏輯清晰、流程自然、模塊化且易於維護，適合學術研究和工程應用。

## 功能特點
- **模塊化設計**：分離數據處理、可視化、模型、損失計算、日誌記錄等模塊。
- **靈活配置**：支持 JSON 配置文件和命令行參數，輕鬆調整超參數。
- **進階損失**：支持 MSE、L1 損失和 L2 正則化，通過配置文件靈活設置。
- **學習率調度**：支持 `StepLR` 和 `ReduceLROnPlateau` 動態調整學習率。
- **錯誤處理與日誌**：全面的錯誤處理和日誌記錄，便於調試。
- **單元測試**：覆蓋數據、損失和模型模塊，確保代碼可靠性。

## 目錄結構
```
pinn_template/
├── config.json        # 配置文件（超參數）
├── data.py            # 數據加載模塊
├── visualize.py       # 可視化模塊
├── logger.py          # 日誌記錄模塊
├── loss.py            # 損失計算模塊
├── model.py           # 模型定義模塊
├── train.py           # 訓練腳本
├── inference.py       # 推理腳本
├── tests/             # 單元測試目錄
│   ├── test_data.py
│   ├── test_loss.py
│   ├── test_model.py
└── README.md          # 項目說明
```

## 環境要求
- Python 3.8+
- PyTorch 1.12+
- NumPy
- SciPy
- Matplotlib
- 詳細依賴見 `requirements.txt`（可選）

## 安裝
1. 克隆項目：
   ```bash
   git clone <repository_url>
   cd pinn_template
   ```
2. 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```
   或使用 Conda：
   ```bash
   conda env create -f environment.yml
   conda activate pinn_env
   ```

## 使用方法
### 1. 配置參數
編輯 `config.json` 修改超參數，例如：
- `N_u`：訓練數據點數
- `layers`：神經網絡結構
- `loss_weights`：損失權重（MSE、L1、正則化）
- `scheduler`：學習率調度參數

### 2. 訓練模型
運行訓練腳本：
```bash
python train.py --N_u 2000 --n_adam 10000 --file_path ../Data/burgers_shock.mat --model_path model.pth
```
支持命令行參數覆蓋 `config.json` 中的默認值。

### 3. 推理與可視化
運行推理腳本：
```bash
python inference.py --file_path ../Data/burgers_shock.mat --model_path model.pth --save_path prediction.png
```
推理結果將生成圖像並保存到指定路徑。

### 4. 運行單元測試
運行所有測試：
```bash
python -m unittest discover tests
```
或使用 pytest：
```bash
pytest tests/
```

## 擴展指南
要將模板應用於其他 PDE 或任務：
1. **修改 PDE**：
   - 在 `model.py` 的 `PINNTrainer.net_f` 中定義新的 PDE 殞差。
   - 可選：將 PDE 提取到獨立模塊（如 `pde.py`）。
2. **添加新損失**：
   - 在 `loss.py` 的 `PINNLoss.compute` 中添加新損失項。
   - 更新 `config.json` 的 `loss_weights`。
3. **自定義數據**：
   - 修改 `data.py` 的 `load_data` 支持新數據格式。
4. **擴展測試**：
   - 在 `tests/` 中添加新測試用例，確保功能正確。

## 聯繫方式
如有問題或建議，請聯繫：<your_email@example.com>

## 許可證
MIT License