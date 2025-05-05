import json
from logger import get_logger

logger = get_logger('Config')

def load_config(config_path='config.json'):
    """加載 JSON 配置文件"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"配置文件加載成功：{config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"配置文件未找到：{config_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"配置文件格式錯誤：{config_path}")
        raise
    except Exception as e:
        logger.error(f"配置文件加載失敗：{str(e)}")
        raise

CONFIG = load_config()