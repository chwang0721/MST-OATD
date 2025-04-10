import os
from functools import partial
from multiprocessing import Pool, Manager

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split
from config import args

from preprocess_utils import *


def preprocess(file, traj_path, data_dir, shortest, longest, boundary, lat_size, lng_size, lng_grid_num, convert_date,
               timestamp_gap, in_boundary, cutting_trajs, traj_nums, point_nums):
    # Read and sort trajectories based on id and timestamp
    try:
        data = pd.read_csv(f"{traj_path}/{file}.txt", header=None)
    except EmptyDataError:
        return
    data.columns = ['id', 'timestamp', 'lon', 'lat']

    trajs = []
    traj_seq = []
    valid = True

    pre_point = data.iloc[0]

    # Select trajectories
    for point in data.itertuples():

        if timestamp_gap(pre_point.timestamp, point.timestamp) <= 1900:
            if in_boundary(point.lat, point.lon, boundary):
                grid_i = int((point.lat - boundary['min_lat']) / lat_size)
                grid_j = int((point.lon - boundary['min_lon']) / lng_size)
                traj_seq.append([grid_i * lng_grid_num + grid_j, convert_date(point.timestamp)])
            else:
                valid = False

        else:
            if valid and len(traj_seq) > 1:
                if shortest <= len(traj_seq) <= longest:
                    trajs.append(traj_seq)
                elif len(traj_seq) > longest:
                    trajs += cutting_trajs(traj_seq, longest, shortest)

            traj_seq = []
            valid = True
        pre_point = point

    if len(trajs) <= 0:
        return

    traj_nums.append(len(trajs))
    point_nums.append(sum([len(traj) for traj in trajs]))
    np.save(f"{data_dir}/data_{file}.npy", np.array(trajs, dtype=object))

# Parallel preprocess
def batch_preprocess(train_data, data_dir):
    manager = Manager()
    traj_nums = manager.list()
    point_nums = manager.list()

    pool = Pool(processes=args.processes)
    pool.map(partial(preprocess, traj_path=traj_path, data_dir=data_dir, shortest=shortest, longest=longest, boundary=boundary,
                     lat_size=lat_size, lng_size=lng_size, lng_grid_num=lng_grid_num, convert_date=convert_date,
                     timestamp_gap=timestamp_gap, in_boundary=in_boundary, cutting_trajs=cutting_trajs,
                     traj_nums=traj_nums, point_nums=point_nums), train_data)

    pool.close()
    pool.join()

    print("Total trajectory num:", sum(traj_nums))
    print("Total point num:", sum(point_nums))


def merge(file_list, path, outfile):
    trajs = []


    for file in file_list:
        try:
            file_trajs = np.load(f"{path}/data_{file}.npy", allow_pickle=True)
            os.remove(f"{path}/data_{file}.npy")
        except FileNotFoundError: # empty files are skipped
            continue
        for traj in file_trajs:
            trajs.append(traj)

    np.save(f"{path}/{outfile}.npy", np.array(trajs, dtype=object))

if __name__ == "__main__":
    np.random.seed(1234)
    traj_path = "../datasets/tdrive"
    data_dir = "../data/tdrive"

    shortest, longest = 0, 250

    boundary = {'min_lat': 0.1, 'max_lat': 90, 'min_lon': 0.1, 'max_lon': 250}

    lat_size, lng_size, lat_grid_num, lng_grid_num = grid_mapping(boundary, args.grid_size)
    print('Grid size:', (lat_grid_num, lng_grid_num))
    A, D = generate_matrix(lat_grid_num, lng_grid_num)

    sparse.save_npz(f'{data_dir}/adj.npz', A)
    sparse.save_npz(f'{data_dir}/d_norm.npz', D)

    print('Preprocessing')

    files = [os.path.splitext(item)[0] for item in os.listdir(traj_path)]
    batch_preprocess(files, data_dir)


    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    print('Merging train trajs')
    merge(test_files, data_dir, "train_init")
    print('Merging test trajs')
    merge(train_files, data_dir, "test_init")

    print('Finished!')
