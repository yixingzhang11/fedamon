import numpy as np
import csv


def csvto1dnumpy(filename):
    data = []
    with open(filename, "r", newline="") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            try:
                data.extend([float(item) for item in row])
            except ValueError:
                print(row)

    return np.array(data)


def splitdata(filename, users):
    geolife = csvto1dnumpy(filename)
    total_length = len(geolife)
    time = len(geolife) // users
    new_total_length = total_length - time * users
    data = geolife[:-new_total_length]
    return data.reshape(users, time)

