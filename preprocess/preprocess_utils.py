import os

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from geopy import distance
from pandas.errors import EmptyDataError

from config import args
from sklearn.model_selection import train_test_split


# Determine whether a point is in boundary
def in_boundary(lat, lng, b):
    return b['min_lon'] < lng < b['max_lon'] and b['min_lat'] < lat < b['max_lat']


# Cut long trajectories
def cut_trajectory(trajectory, longest, shortest):
    trajectories = []
    while len(trajectory) > longest:
        random_length = np.random.randint(shortest, longest)
        trajectories.append(trajectory[:random_length])
        trajectory = trajectory[random_length:]
    if len(trajectory) >= shortest:
        trajectories.append(trajectory)
    return trajectories


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
    g = nx.grid_2d_graph(lat_grid_num, lng_grid_num, periodic=False)
    a = nx.adjacency_matrix(g)
    i = sparse.identity(lat_grid_num * lng_grid_num)
    d = np.diag(np.sum(a + i, axis=1))
    d = 1 / (np.sqrt(d) + 1e-10)
    d[d == 1e10] = 0.
    d = sparse.csr_matrix(d)
    return a + i, d

def create_grid(boundary):
    lat_size, lon_size, lat_grid_num, lon_grid_num = grid_mapping(boundary, args.grid_size)

    print('Grid size:', (lat_grid_num, lon_grid_num))
    a, d = generate_matrix(lat_grid_num, lon_grid_num)

    sparse.save_npz(f'../data/{args.dataset}/adj.npz', a)
    sparse.save_npz(f'../data/{args.dataset}/d_norm.npz', d)

    return lat_size, lon_size, lon_grid_num

def preprocess(file, shortest, longest, boundary, convert_date,
               timestamp_gap, grid_size, traj_nums, point_nums, columns):

    # Unpack
    (lat_size, lon_size, lon_grid_num) = grid_size

    # Read and sort trajectories based on id and timestamp
    try:
        data = pd.read_csv(f"../datasets/{args.dataset}/{file}", header=None, iterator=True, names=columns, chunksize=args.chunk_size)
    except EmptyDataError:
        return
    filename = os.path.splitext(file)[0]

    # Overridden on first iteration
    point_seq = []
    valid = True
    pre_point = None

    trajectories = []
    for chunk in data:
        for point in chunk.itertuples():
            if pre_point and point.id == pre_point.id and timestamp_gap(pre_point.timestamp, point.timestamp) <= args.max_traj_time_delta:
                if in_boundary(point.lat, point.lon, boundary):
                    grid_i = int((point.lat - boundary['min_lat']) / lat_size)
                    grid_j = int((point.lon - boundary['min_lon']) / lon_size)
                    point_seq.append([grid_i * lon_grid_num + grid_j, convert_date(point.timestamp)])
                else:
                    valid = False

            else:
                if valid and len(point_seq) > shortest:
                    if len(point_seq) <= longest:
                        trajectories.append(point_seq)  # add all points as a single trajectory
                    else:
                        trajectories += cut_trajectory(point_seq, longest, shortest) # split point sequence into multiple trajectories
                point_seq = []
                valid = True
            pre_point = point

    if valid and len(point_seq) > shortest:
        if len(point_seq) <= longest:
            trajectories.append(point_seq)  # add all points as a single trajectory
        else:
            trajectories += cut_trajectory(point_seq, longest,
                                           shortest)  # split point sequence into multiple trajectories

    if len(trajectories) <= 0:
        return

    # At the end of your processing loop, before saving:
    trajectories_array = np.ndarray(shape=(len(trajectories),), dtype=object, buffer=np.array(trajectories, dtype=object))



    if len(trajectories_array.shape) != 1:
        print(f"{filename} {trajectories_array.shape}")

    traj_nums.append(len(trajectories))
    point_nums.append(sum([len(traj) for traj in trajectories]))

    np.save(f"../data/{args.dataset}/data_{filename}.npy", trajectories_array)

def merge(files, outfile):
    trajectories = []

    for file in files:
        try:
            filename = os.path.splitext(file)[0]
            file_trajectories = np.load(f"../data/{args.dataset}/data_{filename}.npy", allow_pickle=True)
            trajectories.append(file_trajectories)
        except FileNotFoundError: # empty files are skipped
            continue

    merged_trajectories = np.concatenate(trajectories, axis=0)
    np.save(f"../data/{args.dataset}/{outfile}", merged_trajectories)

# merges 80% of the files into train_init.npy and 20% into test_init.npy
def split_and_merge_files(files):
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    print('Merging train trajectories')
    merge(test_files, "train_init")
    print('Merging test trajectories')
    merge(train_files, "test_init")

    print('Finished!')

def main():
    traj_path = f"../datasets/{args.dataset}"
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
