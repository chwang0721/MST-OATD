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

def grid_mapping_test(boundary, grid_size:float):
    lat_dist = distance.distance((boundary['min_lat'], boundary['min_lon']), 
                                 (boundary['max_lat'], boundary['min_lon'])).km
    lat_size = (boundary['max_lat'] - boundary['min_lat']) / lat_dist * grid_size
    
    lng_dist = distance.distance((boundary['min_lat'], boundary['min_lon']),
                                 (boundary['min_lat'], boundary['max_lon'])).km
    lng_size = (boundary['max_lon'] - boundary['min_lon']) / lng_dist * grid_size

    lat_grid_num = int(lat_dist / grid_size) + 1
    lng_grid_num = int(lng_dist / grid_size) + 1

    return lat_size, lng_size, lat_grid_num, lng_grid_num

boundary = {'min_lat': 30.6, 'max_lat': 30.75, 'min_lon': 104, 'max_lon': 104.16}

aa = grid_mapping_test(boundary, 0.1)


def grid_mapping(boundary, grid_num:Tuple[int,int]):
    lat_grid_num, lng_grid_num = grid_num
    lat_dist = distance.distance((boundary['min_lat'], boundary['min_lon']), 
                                 (boundary['max_lat'], boundary['min_lon'])).km

    lat_size = (boundary['max_lat'] - boundary['min_lat']) / lat_grid_num
    # lat_size = (boundary['max_lat'] - boundary['min_lat']) / lat_dist * 0.1
    # print(lat_size)

    lng_dist = distance.distance((boundary['min_lat'], boundary['min_lon']),
                                 (boundary['min_lat'], boundary['max_lon'])).km

    lng_size = (boundary['max_lon'] - boundary['min_lon']) / lng_grid_num

    # print("lat_dist: ")
    # print(lat_dist)
    # print("lat_size: ")
    # print(lat_size)
    # print("lng_dist: ")
    # print(lng_dist)
    # print("lng_size: ")
    # print(lng_size)

    # print("lat_grid_num: ")
    # print(lat_grid_num)
    # print("lng_grid_num: ")
    # print(lng_grid_num)

    return lat_size, lng_size, lat_grid_num, lng_grid_num

xy = [165, 153]
a = grid_mapping(boundary, xy)

print(aa)
print(a)