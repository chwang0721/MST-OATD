import datetime
import json
import random

from preprocess_utils import *
from config import args

def preprocess(trajectories, shortest, longest, grid_size, boundary):
    processed_trajectories = []

    traj_num, point_num = 0, 0

    (lat_size, lon_size, lon_grid_num) = grid_size

    for traj in trajectories.itertuples():

        traj_seq = []
        valid = True  # Flag to determine whether a trajectory is in boundary

        polyline = json.loads(traj.POLYLINE)
        timestamp = traj.TIMESTAMP

        if len(polyline) < shortest:
            continue

        for lng, lat in polyline:

            if in_boundary(lat, lng, boundary):
                grid_i = int((lat - boundary['min_lat']) / lat_size)
                grid_j = int((lng - boundary['min_lon']) / lon_size)

                t = datetime.datetime.fromtimestamp(timestamp)
                t = [t.hour, t.minute, t.second, t.year, t.month, t.day]  # Time vector

                traj_seq.append([int(grid_i * lon_grid_num + grid_j), t])
                timestamp += 15  # In porto dataset, the sampling rate is 15

            else:
                valid = False
                break

        # Randomly delete 30% trajectory points to make the sampling rate not fixed
        to_delete = set(random.sample(range(len(traj_seq)), int(len(traj_seq) * 0.3)))
        traj_seq = [item for index, item in enumerate(traj_seq) if index not in to_delete]

        # Lengths are limited from 20 to 50
        if valid:
            if len(traj_seq) <= longest:
                processed_trajectories.append(traj_seq)
            else:
                processed_trajectories += cut_trajectory(traj_seq, longest, shortest)

    traj_num += len(processed_trajectories)

    for traj in processed_trajectories:
        point_num += len(traj)

    return processed_trajectories, traj_num, point_num

def time_convert(timestamp):
    return datetime.datetime.fromtimestamp(timestamp)

def main():
    random.seed(1234)
    np.random.seed(1234)

    boundary = {'min_lat': 41.140092, 'max_lat': 41.185969, 'min_lon': -8.690261, 'max_lon': -8.549155}
    shortest, longest = 20, 50

    grid_size = create_grid(boundary)
    print('Preprocessing Porto')

    # Read csv file
    file = "porto.csv"
    trajectories = pd.read_csv(f"../datasets/{args.dataset}/{file}", header=0, usecols=['POLYLINE', 'TIMESTAMP'])
    trajectories['TIMESTAMP'].apply(time_convert)

    # Initial dataset
    start_time = datetime.datetime(2013, 7, 1, 0, 0, 0)
    end_time = datetime.datetime(2013, 9, 1, 0, 0, 0)

    # Select trajectories from start time to end time
    bounded_trajectories = trajectories[start_time <= trajectories['TIMESTAMP'] < end_time]
    preprocessed_trajectories, traj_num, point_num = preprocess(bounded_trajectories, shortest, longest, grid_size, boundary)
    train_data, test_data = train_test_split(preprocessed_trajectories, test_size=0.2, random_state=42)

    np.save(f"../data/{args.dataset}/train_data_init.npy", np.array(train_data, dtype=object))
    np.save(f"../data/{args.dataset}/test_data_init.npy", np.array(test_data, dtype=object))

    start_time = datetime.datetime(2013, 9, 1, 0, 0, 0)

    # Evolving dataset
    for month in range(1, 11):
        end_time = start_time + datetime.timedelta(days=30)
        bounded_trajectories = trajectories[start_time <= trajectories['TIMESTAMP'] < end_time]

        preprocessed_trajectories, traj_numm, point_numm = preprocess(bounded_trajectories, shortest, longest, grid_size, boundary)
        traj_num += traj_numm
        point_numm += point_numm
        train_data, test_data = train_test_split(preprocessed_trajectories, test_size=0.2, random_state=42)

        np.save(f"../data/{args.dataset}/train_data_{month}.npy", np.array(train_data, dtype=object))
        np.save(f"../data/{args.dataset}/test_data_{month}.npy", np.array(test_data, dtype=object))

        start_time = end_time

    # Dataset statistics
    print("Total trajectory num:", traj_num)
    print("Total point num:", point_num)

    print('Finished!')
