# logger.py
# ------------------------------------------------------------
# 功能：
#   1. 初始化並返回一個名為 name 的日誌器實例
#   2. 從 Config 讀取日誌級別、文件路徑等配置
#   3. 配置文件處理器（RotatingFileHandler）以管理日誌文件大小和備份
#   4. 可選配置控制台處理器以輸出日誌到終端
# ------------------------------------------------------------
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from config import Config

# 創建並返回一個日誌器
def get_logger(name:str)->logging.Logger: 

    logger = logging.getLogger(name)

    # 若有處理器，直接返回，避免重複配置
    if logger.handlers:
        return logger
    
    # ========== 複用Config實例 =========== #
    config_instance  = Config()

    # ========== 等級 =========== #
    try:
        level_str = config_instance.get('log_level', 'INFO').upper()
        # 反射函數，獲取屬性值
        level = getattr(logging, level_str)
    except AttributeError:
        # 若配置的日志级别无效，使用默认的 INFO 级别
        level = logging.INFO
        logging.warning(f"Invalid log level '{level_str}' in config. Using INFO instead.")
    logger.setLevel(level)

    # ========== 格式 =========== #
    fmt_file = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    fmt_console = '%(name)s - %(levelname)s - %(message)s'
    fmt_date = '%Y-%m-%d %H:%M:%S'

    # ========== 文件handler =========== #
    # expanduser() 方法的作用是展開路徑中的 ~ 符號，將其替換為當前用戶的主目錄路徑。
    # resolve() 方法用於解析路徑，將相對路徑轉換為絕對路徑，同時解析路徑中的符號鏈接（symbolic links）。
    # 它會處理路徑中的 .（當前路徑）、..（上一級目錄）等特殊符號，最終返回一個絕對路徑。
    log_path = Path(config_instance.get('log_file','app.log')).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    """
    RotatingFileHandler 能在日志文件达到指定大小时，自动创建新的日志文件，并保留一定数量的旧日志文件。
    """
    fh = RotatingFileHandler(filename=log_path,
                             # 最大文件大小，單位為字节，默認為10MB
                             maxBytes=config_instance.get('log_max_bytes',1024*1024*10),
                             # 舊日誌文件的最大數量，默認為5
                             backupCount=config_instance.get('backup_count',5),
                             # 文件編碼，默認為utf-8
                             encoding=config_instance.get('log_encoding','utf-8')
                             )
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt_file,fmt_date))
    logger.addHandler(fh)

    # ========== 控制台handler =========== #
    if config_instance.get('log_console',True):
        # 創建一個流處理器，用於將日誌輸出到終端
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter(fmt_console,fmt_date))
        logger.addHandler(ch)
    
    logger.propagate = False
    return logger

# 示例用法：
# logger = get_logger(__name__)
# logger.info("This is an info message.")