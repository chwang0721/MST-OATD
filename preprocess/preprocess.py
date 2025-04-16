import numpy as np

from config import args

from preprocess_tdrive import main as tdrive
from preprocess_porto import main as porto
from preprocess_cd import main as cd
from preprocess_utils import main as util

if __name__ == '__main__':
    np.random.seed(1234)
    if args.find_boundary:
        util()
    else:
        match args.dataset:
            case "tdrive":
                tdrive()
            case "porto":
                porto()
            case "cd":
                cd()


