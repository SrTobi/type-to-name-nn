#!python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from src import IdentifierPredictor

predictor = IdentifierPredictor("all_vals", log = True)
print("Run training...")
predictor.train(save_every_x_epoch = 1000)