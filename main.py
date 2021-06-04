from config import config
from utils import set_seed, run, inference
from data import Tokenizer

set_seed(seed=config.general.seed)

if not config.inference.inference:
    run(config)
else:
    inference(config) # Not implemented, inference was computed on Kaggle 
