import os
import sys
import logging
from tkinter.tix import Tree

logging_str = "[%(asctime)s: %(levlename)s: %(module)s: %(message)s"
log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir,exist_ok=True)

logging.basicConfig(
    level=logging.info,
    format = logging_str,
    handlers= [logging.StreamHandler(sys.stdout),
    logging.FileHandler(log_filepath)]
    
)

logger = logging.getLogger("deepClassifier")