import os
from yacs.config import CfgNode as CN

"""
This file only used for default settings, please further specify the configurations in `exp.yaml`.
"""
_C = CN()

# -----------------------------------------------------------------------------
# Eexperiment
# -----------------------------------------------------------------------------
_C.EXP = CN()
_C.EXP.NAME = "test"
_C.EXP.DEVICE = "cuda"

# -----------------------------------------------------------------------------
# Path setting
# -----------------------------------------------------------------------------
_C.PATH = CN()
_C.PATH.CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_C.PATH.ROOT_DIR = os.path.dirname(_C.PATH.CONFIG_DIR)
_C.PATH.DATA_DIR = os.path.join(_C.PATH.ROOT_DIR, 'data')
_C.PATH.TRAIN_FILE = os.path.join(_C.PATH.DATA_DIR, "train_subset.json")
_C.PATH.VALID_FILE = os.path.join(_C.PATH.DATA_DIR, "valid_subset.json")
_C.PATH.TEST_FILE = os.path.join(_C.PATH.DATA_DIR, "test_subset.json")
_C.PATH.VOCAB_FILE = os.path.join(_C.PATH.DATA_DIR, "vocab/")
_C.PATH.PRETRAINED_EMBEDDING_FILE = os.path.join(_C.PATH.DATA_DIR, "glove.twitter.27B.200d.txt")
_C.PATH.LOG_DIR = os.path.join(_C.PATH.ROOT_DIR, "saved/" + _C.EXP.NAME + "/")

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.SENTENCE = 8 # only set for hier models
_C.DATA.MAX_SEQ_LENGTH = 200
_C.DATA.VOCAB_MIN_COUNT = 20
_C.DATA.SENTENCE_SEGMENTATION = False

# -----------------------------------------------------------------------------
# Model (specific model setting to-do in yaml config files)
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = "NNClassifier"
_C.MODEL.EMBEDDING_PRETRAINED = True
_C.MODEL.BIDIRECTIONAL = False

# -----------------------------------------------------------------------------
# Feedforward settings
# -----------------------------------------------------------------------------
_C.FEED = CN()
_C.FEED.INPUT_SIZE =  512
_C.FEED.LAYERS = 2
_C.FEED.HIDDEN = [256,100]
_C.FEED.DROPOUT2 = 0
# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------
_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 2
_C.SYSTEM.DEVICE_ID = 0

# -----------------------------------------------------------------------------
# Architecture (For general architecture, e.g. encoder hidden dim)
# -----------------------------------------------------------------------------
_C.ARCH = CN()
_C.ARCH.EMBEDDING = CN()
_C.ARCH.EMBEDDING.DIM = 200

_C.ARCH.SENTENCE_ENCODER = CN()
_C.ARCH.SENTENCE_ENCODER.TYPE = "lstm"
_C.ARCH.SENTENCE_ENCODER.INPUT_DIM = 200
_C.ARCH.SENTENCE_ENCODER.HIDDEN_DIM = 256
_C.ARCH.SENTENCE_ENCODER.NUM_LAYER = 1
_C.ARCH.SENTENCE_ENCODER.TRANSFORMER_NHEAD = 5
_C.ARCH.SENTENCE_ENCODER.BIDIR = True
_C.ARCH.SENTENCE_ENCODER.DROPOUT = 0
_C.ARCH.SENTENCE_ENCODER.FILTER_NUM = 3
_C.ARCH.SENTENCE_ENCODER.OUTPUT_DIM = 2 

_C.ARCH.HAN = CN()
_C.ARCH.HAN.GRU_HIDDEN = 100

# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.PATIENCE = 3
_C.TRAIN.NUM_EPOCHS = 50
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.LR = 1e-5
_C.TRAIN.L2_NORM = 1e-5
_C.TRAIN.GRAD_NORM = 2.0
_C.TRAIN.GRAD_CLIPPING = 2.0
_C.TRAIN.EPS = 1e-08
# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
_C.PREDICT = CN()
_C.PREDICT.PREDICT_FILE = os.path.join(_C.PATH.DATA_DIR, "test_data.json")

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
