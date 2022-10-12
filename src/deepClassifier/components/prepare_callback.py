from deepClassifier.entity import PrepareCallBackConfig
from deepClassifier import logger
from deepClassifier.utils import get_size
from tqdm import tqdm
from pathlib import Path
from urllib import request
import os
from zipfile import ZipFile
import time
import tensorflow as tf


class PrepareCallBack:
    def __init__(self,config:PrepareCallBackConfig):
        self.config=config

    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}"
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    
    @property
    def _create_ckpt_callbacks(self):
       
        return tf.keras.callbacks.ModelCheckpoint(
            filepath= self.config.checkpoint_model_filepath,
            save_best_only= True
            )

        
    def get_call_back(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]