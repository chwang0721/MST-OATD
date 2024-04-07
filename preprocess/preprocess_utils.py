import datetime
import time

import networkx as nx
import numpy as np
import scipy.sparse as sparse
from geopy import distance


# Determine whether a point is in boundary
def in_boundary(lat, lng, b):
    return b['min_lng'] < lng < b['max_lng'] and b['min_lat'] < lat < b['max_lat']


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
    timeArray = time.strptime(str, "%Y/%m/%d %H:%M:%S")
    t = [timeArray.tm_hour, timeArray.tm_min, timeArray.tm_sec, timeArray.tm_year, timeArray.tm_mon, timeArray.tm_mday]
    return t


# Calculate timestamp gap
def timestamp_gap(str1, str2):
    timestamp1 = datetime.datetime.strptime(str1, "%Y/%m/%d %H:%M:%S")
    timestamp2 = datetime.datetime.strptime(str2, "%Y/%m/%d %H:%M:%S")
    return (timestamp2 - timestamp1).total_seconds()


# Map trajectories to grids
def grid_mapping(boundary, grid_size):
    lat_dist = distance.distance((boundary['min_lat'], boundary['min_lng']),
                                 (boundary['max_lat'], boundary['min_lng'])).km
    lat_size = (boundary['max_lat'] - boundary['min_lat']) / lat_dist * grid_size

    lng_dist = distance.distance((boundary['min_lat'], boundary['min_lng']),
                                 (boundary['min_lat'], boundary['max_lng'])).km
    lng_size = (boundary['max_lng'] - boundary['min_lng']) / lng_dist * grid_size

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
