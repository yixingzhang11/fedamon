import numpy as np
from datasets.geolife.generate import *


def read_data_transpose(file_path):
    data = np.genfromtxt(file_path, delimiter=",")
    return data.T


def read_data(file_path):
    data = np.genfromtxt(file_path, delimiter=",")
    return data


def read(file_path):
    if file_path == "../fedamon/datasets/ericsson/cpu.csv":
        data = read_data_transpose(file_path)
    elif file_path == "../fedamon/datasets/ucr/power.csv":
        data = read_data(file_path)
    elif file_path == "../fedamon/datasets/geolife/speed.csv":
        data = splitdata(file_path, users=200)
    elif file_path == "../fedamon/datasets/intelLab/temp.csv":
        data = read_data(file_path)
    else:
        raise FileNotFoundError
    return data
