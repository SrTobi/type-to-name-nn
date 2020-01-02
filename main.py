#!python
import os
import sys
import argparse


parser = argparse.ArgumentParser(description="Predicts identifiers.")
parser.add_argument("expr", type=str, default=None, nargs="?",
                    help="An expression for which to predict the identifier.")
parser.add_argument("-t", "--train", dest="train", action="store_true",
                    help="Start the training")
parser.add_argument("--tflog", dest="tf_level", type=int, default=3,
                    choices=[0, 1, 2, 3],
                    help="Tensorflow log level (0, 1, 2, or 3)")
parser.add_argument("-q", "--quiet", dest="log", action="store_false",
                    help="Don't print log")

args = parser.parse_args()

if args.train == (args.expr is not None):
    parser.print_help()
    print()
    print("Either specify an expression or --train")
    sys.exit()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(args.tf_level)


from src import IdentifierPredictor

predictor = IdentifierPredictor("all_vals", log = args.log)

predictor.load(partial = not args.train)

if args.train:
    predictor.log("Run training...")
    predictor.train(save_every_x_epoch = 1)
elif args.expr is not None:
    predictor.evaluate("test")