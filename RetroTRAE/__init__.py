#import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import json
from .utils import *
from .predict import *
from .database import *

configs = json.load(open(ckpt_dir.joinpath('configs.json')))
try:
    configs = json.load(open(ckpt_dir.joinpath('configs.json')))
except:
    pass

