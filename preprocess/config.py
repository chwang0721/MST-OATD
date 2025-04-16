import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--processes", type=int, default=10)
parser.add_argument("--grid_size", type=float, default=0.1)
parser.add_argument("--dataset", type=str, default='porto')
parser.add_argument("--max_traj_time_delta", type=int, default=1900)
parser.add_argument("--find_boundary", type=bool, default=False)



args = parser.parse_args()
