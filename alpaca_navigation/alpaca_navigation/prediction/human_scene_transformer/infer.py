"""
from skeleton and position to trajectory prediction
"""

import collections
import os
from typing import Sequence


import numpy as np
import tensorflow as tf
import tqdm
import time
# import gin

from alpaca_navigation.prediction.utils.logger import logger
from alpaca_navigation.prediction.prediction_config import *
# from alpaca_navigation.prediction.human_scene_transformer.jrdb import dataset_params
# from alpaca_navigation.prediction.human_scene_transformer.jrdb import input_fn
# from alpaca_navigation.prediction.human_scene_transformer.metrics import metrics
from alpaca_navigation.prediction.human_scene_transformer.model import model as hst_model
from alpaca_navigation.prediction.human_scene_transformer.model import model_params

from alpaca_navigation.prediction.human_scene_transformer.config.hst_config import hst_dataset_param, hst_model_param, TEST_SCENES

_checkpoint_path = HST_CKPT_PATH.as_posix()
_param_path = NETWORK_PARAM_PATH.as_posix()


def init_model() -> hst_model.HumanTrajectorySceneTransformer:
    logger.info(f"\nhst model param")
    for key, value in vars(hst_model_param).items():
        logger.info(f"{key}:   {value}")
    
    model = hst_model.HumanTrajectorySceneTransformer(hst_model_param)
    return model


# def eval_human_traj():
    



if __name__ == "__main__":
    # export TF_ENABLE_ONEDNN_OPTS=0
    # export TF_DISABLE_MKL=1
    # export PYTHONPATH=/home/user/ros2_ws/src/hst_node:$PYTHONPATH
    init_model()