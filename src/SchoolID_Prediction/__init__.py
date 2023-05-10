import logging
from pathlib import Path

logging_file_path=Path("ruuning_logs.log")
format_str=f" [ %(asctime)s    %(filename)s    %(funcName)s ]   [ %(message)s ]"
logging.basicConfig(
    filename=logging_file_path,
    level=logging.INFO,
    format=format_str
)