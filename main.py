#!python
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from src import IdentifierPredictor

predictor = IdentifierPredictor("learn_simple.txt", log = True)
print("Run training...")
predictor.train()