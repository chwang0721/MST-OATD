import datetime
import os
import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from geopy import distance
from pandas.errors import EmptyDataError


# Determine whether a point is in boundary
def in_boundary(lat, lng, b):
    return b['min_lon'] < lng < b['max_lon'] and b['min_lat'] < lat < b['max_lat']


# Cut long trajectories
def cutting_trajs(traj, longest, shortest):
    cutted_trajs = []
    while len(traj) > longest:
        random_length = np.random.randint(shortest, longest)
        cutted_traj = traj[:random_length]
        cutted_trajs.append(cutted_traj)
        traj = traj[random_length:]
    return cutted_trajs


# convert datetime to time vector
def convert_date(str):
    timeArray = time.strptime(str, "%Y-%m-%d %H:%M:%S")
    t = [timeArray.tm_hour, timeArray.tm_min, timeArray.tm_sec, timeArray.tm_year, timeArray.tm_mon, timeArray.tm_mday]
    return t


# Calculate timestamp gap
def timestamp_gap(str1, str2):
    timestamp1 = datetime.datetime.strptime(str1, "%Y-%m-%d %H:%M:%S")
    timestamp2 = datetime.datetime.strptime(str2, "%Y-%m-%d %H:%M:%S")
    return (timestamp2 - timestamp1).total_seconds()


# Map trajectories to grids
def grid_mapping(boundary, grid_size):
    lat_dist = distance.distance((boundary['min_lat'], boundary['min_lon']),
                                 (boundary['max_lat'], boundary['min_lon'])).km
    lat_size = (boundary['max_lat'] - boundary['min_lat']) / lat_dist * grid_size

    lng_dist = distance.distance((boundary['min_lat'], boundary['min_lon']),
                                 (boundary['min_lat'], boundary['max_lon'])).km
    lng_size = (boundary['max_lon'] - boundary['min_lon']) / lng_dist * grid_size

    lat_grid_num = int(lat_dist / grid_size) + 1
    lng_grid_num = int(lng_dist / grid_size) + 1
    return lat_size, lng_size, lat_grid_num, lng_grid_num


# Generate adjacency matrix and normalized degree matrix
def generate_matrix(lat_grid_num, lng_grid_num):
    G = nx.grid_2d_graph(lat_grid_num, lng_grid_num, periodic=False)
    A = nx.adjacency_matrix(G)
    I = sparse.identity(lat_grid_num * lng_grid_num)
    D = np.diag(np.sum(A + I, axis=1))
    D = 1 / (np.sqrt(D) + 1e-10)
    D[D == 1e10] = 0.
    D = sparse.csr_matrix(D)
    return A + I, D


if __name__ == '__main__':
    traj_path = "../datasets/tdrive"
    min_lat = [float("inf")]
    max_lat = [-float("inf")]

    min_lon = [float("inf")]
    max_lon = [-float("inf")]

    boundary = {'min_lat': 0.1, 'max_lat': 100, 'min_lon': 0.1, 'max_lon': 250}

    path_list = os.listdir(traj_path)
    a = len(path_list)
    x = 0
    lat_maxima_min = []
    lat_maxima_max = []
    lon_maxima_min = []
    lon_maxima_max = []
    for path in path_list:
        print(f"{x}/{a}")
        x = x+1
        try:
            data = pd.read_csv("{}/{}".format(traj_path, path), header=None)
        except EmptyDataError:
            continue
        data.columns = ['id', 'timestamp', 'lon', 'lat']

        for point in data.itertuples():
            if not in_boundary(point.lat, point.lon, boundary):
                break
        else:
            if min_lat > data['lat'].min():
                min_lat = data['lat'].min()
                lat_maxima_min.append(f"{path}: {data['lat'].min()}")
            if max_lat < data['lat'].max():
                max_lat = data['lat'].max()
                lat_maxima_max.append(f"{path}: {data['lat'].max()}")

            if min_lon > data['lon'].min():
                min_lon = data['lon'].min()
                lon_maxima_min.append(f"{path}: {data['lon'].min()}")

            if max_lon < data['lon'].max():
                max_lon = data['lon'].max()
                lon_maxima_max.append(f"{path}: {data['lon'].max()}")

    lat_maxima_min.reverse()
    lat_maxima_max.reverse()
    lon_maxima_min.reverse()
    lon_maxima_max.reverse()
    print(f"lat min: {min_lat}, {lat_maxima_min}")
    print(f"lat max: {max_lat}, {lat_maxima_max}")
    print(f"lon min: {min_lon}, {lon_maxima_min}")
    print(f"lon max: {max_lon}, {lon_maxima_max}")
