import os
from functools import partial
from multiprocessing import Pool, Manager

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split

from preprocess_utils import *


def preprocess(file, traj_path, shortest, longest, boundary, lat_size, lng_size, lng_grid_num, convert_date,
               timestamp_gap, in_boundary, cutting_trajs, traj_nums, point_nums):
    np.random.seed(1234)
    # Read and sort trajectories based on id and timestamp
    data = pd.read_csv("{}/{}".format(traj_path, file), header=None)
    data.columns = ['id', 'lat', 'lon', 'state', 'timestamp']
    data = data.sort_values(by=['id', 'timestamp'])
    data = data[data['state'] == 1]

    trajs = []
    traj_seq = []
    valid = True

    pre_point = data.iloc[0]

    # Select trajectories
    for point in data.itertuples():

        if point.id == pre_point.id and timestamp_gap(pre_point.timestamp, point.timestamp) <= 20:
            if in_boundary(point.lat, point.lon, boundary):
                grid_i = int((point.lat - boundary['min_lat']) / lat_size)
                grid_j = int((point.lon - boundary['min_lng']) / lng_size)
                traj_seq.append([grid_i * lng_grid_num + grid_j, convert_date(point[5])])
            else:
                valid = False

        else:
            if valid:
                if shortest <= len(traj_seq) <= longest:
                    trajs.append(traj_seq)
                elif len(traj_seq) > longest:
                    trajs += cutting_trajs(traj_seq, longest, shortest)

            traj_seq = []
            valid = True
        pre_point = point

    traj_nums.append(len(trajs))
    point_nums.append(sum([len(traj) for traj in trajs]))

    train_data, test_data = train_test_split(trajs, test_size=0.2, random_state=42)
    np.save("../data/cd/train_data_{}.npy".format(file[:8]), np.array(train_data, dtype=object))
    np.save("../data/cd/test_data_{}.npy".format(file[:8]), np.array(test_data, dtype=object))


# Parallel preprocess
def batch_preprocess(path_list):
    manager = Manager()
    traj_nums = manager.list()
    point_nums = manager.list()
    pool = Pool(processes=10)
    pool.map(partial(preprocess, traj_path=traj_path, shortest=shortest, longest=longest, boundary=boundary,
                     lat_size=lat_size, lng_size=lng_size, lng_grid_num=lng_grid_num, convert_date=convert_date,
                     timestamp_gap=timestamp_gap, in_boundary=in_boundary, cutting_trajs=cutting_trajs,
                     traj_nums=traj_nums, point_nums=point_nums), path_list)
    pool.close()
    pool.join()

    num_trajs = sum(traj_nums)
    num_points = sum(point_nums)
    print("Total trajectory num:", num_trajs)
    print("Total point num:", num_points)


def merge(path_list):
    res_train = []
    res_test = []

    for file in path_list:

        train_trajs = np.load("../data/cd/train_data_{}.npy".format(file[:8]), allow_pickle=True)
        test_trajs = np.load("../data/cd/test_data_{}.npy".format(file[:8]), allow_pickle=True)
        res_train.append(train_trajs)
        res_test.append(test_trajs)

    res_train = np.concatenate(res_train, axis=0)
    res_test = np.concatenate(res_test, axis=0)

    return res_train, res_test


def main():
    path_list = os.listdir(traj_path)
    path_list.sort(key=lambda x: x.split('.'))
    path_list = path_list[:10]

    batch_preprocess(path_list)
    train_data, test_data = merge(path_list[:3])

    np.save("../data/cd/train_data_init.npy", np.array(train_data, dtype=object))
    np.save("../data/cd/test_data_init.npy", np.array(test_data, dtype=object))

    print('Fnished!')


if __name__ == "__main__":
    traj_path = "../../../data/wch/datasets/chengdu"

    grid_size = 0.1
    shortest, longest = 30, 100
    boundary = {'min_lat': 30.6, 'max_lat': 30.75, 'min_lng': 104, 'max_lng': 104.16}

    lat_size, lng_size, lat_grid_num, lng_grid_num = grid_mapping(boundary, grid_size)
    A, D = generate_matrix(lat_grid_num, lng_grid_num)

    sparse.save_npz('../data/cd/adj.npz', A)
    sparse.save_npz('../data/cd/d_norm.npz', D)

    print('Grid size:', (lat_grid_num, lng_grid_num))
    print('----------Preprocessing----------')
    main()
