import os
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from geopy import distance
from pandas.errors import EmptyDataError

from config import args
from sklearn.model_selection import train_test_split
import argparse

def float_or_int_tuple(value):
    """Custom type function to accept either float or tuple of ints."""
    try:
        # Case 1: Try parsing as a float
        return float(value)
    except ValueError:
        try:
            # Case 2: Try parsing as a tuple of ints (e.g., "1,2")
            # Split by commas and convert to integers
            parts = [int(x.strip()) for x in value.split(',')]
            return tuple(parts)
        except (ValueError, AttributeError):
            raise argparse.ArgumentTypeError(
                f"Must be a float or tuple of ints (e.g., '3.14' or '2,3'), got '{value}'"
            )

parser = argparse.ArgumentParser()
parser.add_argument("--grid_size", type=float_or_int_tuple, 
                    help="Input value (float or tuple of ints, standart values for porto e.g., '0.1' or '167,154')", default=0.1)

args = parser.parse_args()

def grid_mapping(boundary, grid_size:float):
    print("IM A FLOAT")
    lat_dist = distance.distance((boundary['min_lat'], boundary['min_lon']), 
                                 (boundary['max_lat'], boundary['min_lon'])).km
    lat_size = (boundary['max_lat'] - boundary['min_lat']) / lat_dist * grid_size
    
    lng_dist = distance.distance((boundary['min_lat'], boundary['min_lon']),
                                 (boundary['min_lat'], boundary['max_lon'])).km
    lng_size = (boundary['max_lon'] - boundary['min_lon']) / lng_dist * grid_size

    lat_grid_num = int(lat_dist / grid_size) + 1
    lng_grid_num = int(lng_dist / grid_size) + 1

    return lat_size, lng_size, lat_grid_num, lng_grid_num

def manuel_grid_mapping(boundary, grid_num:Tuple[int,int]):
    print("IM A TUPLE INT")
    lat_grid_num, lng_grid_num = grid_num
    lat_dist = distance.distance((boundary['min_lat'], boundary['min_lon']), 
                                 (boundary['max_lat'], boundary['min_lon'])).km

    lat_size = (boundary['max_lat'] - boundary['min_lat']) / lat_grid_num
    # lat_size = (boundary['max_lat'] - boundary['min_lat']) / lat_dist * 0.1
    # print(lat_size)

    lng_dist = distance.distance((boundary['min_lat'], boundary['min_lon']),
                                 (boundary['min_lat'], boundary['max_lon'])).km

    lng_size = (boundary['max_lon'] - boundary['min_lon']) / lng_grid_num

    return lat_size, lng_size, lat_grid_num, lng_grid_num

def main():
    boundary = {'min_lat': 41.140092, 'max_lat': 41.185969, 'min_lon': -8.690261, 'max_lon': -8.549155}

    if isinstance(args.grid_size, float):
        grid_mapping(boundary, args.grid_size)
    else:
        manuel_grid_mapping(boundary, args.grid_size)

main()