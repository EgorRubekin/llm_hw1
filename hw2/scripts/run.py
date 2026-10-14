import sys

import train
from experiments import EXPERIMENTS

name = sys.argv[1]
changes = dict(EXPERIMENTS[name])
train.TRAIN_SECONDS = changes.pop("seconds")
train.ATTENTION_IMPLEMENTATION = changes.pop("attention", "sdpa")
train.TRAINING_CONFIG.update(changes, output_dir=f"./results/{name}")

train.main()
