import datetime
import time
from functools import partial
from multiprocessing import Pool, Manager

from preprocess_utils import *
from config import args

# convert datetime to time vector
def convert_date(date_str):
    time_array = time.strptime(date_str, "%Y/%m/%d %H:%M:%S")
    t = [time_array.tm_hour, time_array.tm_min, time_array.tm_sec, time_array.tm_year, time_array.tm_mon, time_array.tm_mday]
    return t


# Calculate timestamp gap
def timestamp_gap(str1, str2):
    timestamp1 = datetime.datetime.strptime(str1, "%Y/%m/%d %H:%M:%S")
    timestamp2 = datetime.datetime.strptime(str2, "%Y/%m/%d %H:%M:%S")
    return (timestamp2 - timestamp1).total_seconds()

def main():
    print('Preprocessing Chengdu')
    files = os.listdir(f"../datasets/{args.dataset}")

    boundary = {'min_lat': 30.6, 'max_lat': 30.75, 'min_lon': 104, 'max_lon': 104.16}
    columns = ['id', 'lat', 'lon', 'state', 'timestamp']
    grid_size = create_grid(boundary)

    manager = Manager()
    traj_nums = manager.list()
    point_nums = manager.list()

    pool = Pool(args.processes)
    pool.map(partial(preprocess, shortest=30, longest=100, boundary=boundary, convert_date=convert_date,
                     timestamp_gap=timestamp_gap, traj_nums=traj_nums, point_nums=point_nums, grid_size=grid_size, columns=columns), files)

    pool.close()
    pool.join()

    print("Total trajectory num:", sum(traj_nums))
    print("Total point num:", sum(point_nums))

    split_and_merge_files(files)
