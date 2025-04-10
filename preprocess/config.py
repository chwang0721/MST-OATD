import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--processes", type=int, default=10)
parser.add_argument("--grid_size", type=float, default=0.1)

args = parser.parse_args()
