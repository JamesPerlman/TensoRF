from argparse import ArgumentParser

from torch import Argument

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--project", type=str, required=True, help="Path to project.")
    parser.add_argument("--steps", type=int, default=10000, help="Number of steps to stop training.")
    parser.add_argument("--load", type=str, help="Path to a checkpoint to load and start training from.")
    parser.add_argument("--output", type=str, default=None, help="Output file path (will be set automatically if None).")

    return parser.parse_args()

# main
if __name__ == "__main__":
    args = parse_args()
    