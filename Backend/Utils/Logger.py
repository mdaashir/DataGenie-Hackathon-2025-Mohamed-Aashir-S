import os
import logging
from Backend import LOGS_DIR

def setup_logging(output_dir=LOGS_DIR, log_name="backend.log"):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_name)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
        ],
    )
    logging.info(f"{log_name.split('.')[0]} logging initialized.")
    return log_path, logging