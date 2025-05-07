# config.py
# ------------------------------------------------------------
# 功能：
#   1. 從指定路徑加載Json配置文件
#   2. 提供鍵值對的存取方法，支持默認值
#   3. 將更新後的配置數據保存回文件 / JSON
#   4. 使用臨時文件確保保存過程安全
# ------------------------------------------------------------
import json
import os
from logger import get_logger

logger = get_logger('Config')

class Config:
    # 配置類，用於讀取和寫入配置文件
    def __init__(self, config_path = 'config.json'):
        self.logger = logger
        self.config_path = config_path
        self.data = {}
        self.load()
    
    # 加載Json配置文件
    def load(self) -> None:
        try:
            with open(self.config_path, 'r') as f:
                self.data = json.load(f)
            self.logger.info(f"配置文件加載成功：{self.config_path}")
        except FileNotFoundError:
            self.logger.warning(f"配置文件不存在：{self.config_path}")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"配置文件格式錯誤：{self.config_path}")
            raise
        except Exception as e:
            self.logger.error(f"配置文件加載失敗：{self.config_path},錯誤信息：{e}")
            raise
    
    # 獲取配置值，支持默認值
    def get(self, key:str, default = None):
        if key not in self.data:
            self.logger.warning(f"配置鍵不存在：{key}")
        return self.data.get(key, default)
    
    # 設置配置值
    def set(self, key, value):
        self.data[key] = value
        self.logger.info(f"配置設置：{key} = {value}")
        self.save()

    # 保存配置文件
    def save(self) -> None:
        # 生成臨時文件，確保保存的是完整的配置
        tmp_path = self.config_path.with_suffix('.json.tmp')
        try:
            with tmp_path.open('w', encoding='utf-8') as f:
                # 將配置文件以Json格式寫入臨時文件
                json.dump(self.data, f, indent=4, ensure_ascii=False)
            # 使用os.replace來原子地替換文件
            os.replace(tmp_path, self.config_path)   
            self.logger.info(f"配置文件保存成功：{self.config_path}")
        except Exception as e:
            self.logger.error(f"配置文件保存失败：{e}")
            # 刪除臨時文件，missing_ok=True表示如果文件不存在也不會報錯
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    # 更新配置值
    def update(self, key, value):
        self.data[key] = value
        self.logger.info(f"配置更新：{key} = {value}")

    