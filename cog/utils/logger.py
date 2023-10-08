import logging
import os

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper() 

logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()
