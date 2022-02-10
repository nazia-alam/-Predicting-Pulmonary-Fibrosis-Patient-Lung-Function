#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# This program takes patient data and applies KNN with DTW to predict lung function

import csv
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

import math
from fastdtw import fastdtw

max_fvc = 6399
min_fvc = 827

diff = max_fvc - min_fvc


def closest(lst, K):
    return min(range(len(lst)), key=lambda i: abs(lst[i]-K))

# Calculates the weighted distance between two patients


def dist(patient1, patient2):
    patient1_series = [[time[0], time[1]] for time in patient1[0]][:-3]
    patient2_series = [[time[0], time[1]] for time in patient2[0]]

    # calculate DTW
    seq_dist, _ = fastdtw(patient1_series, patient2_series)
    binary_dist = 0

    for i in range(1, len(patient1)):
        binary_dist += math.fabs(patient1[i] - patient2[i])

    return seq_dist + binary_dist

# Perform k-nearest neighbors with a given patient


def knn(patient, records, n):
    neighbors = {}

    for i in range(n):
        neighbors[dist(patient, records[i])] = records[i]

    for i in range(n, len(records)):
        pot_dist = dist(patient, records[i])

        max = list(neighbors.keys())[0]

        for distance in neighbors.keys():
            if distance > max:
                max = distance

        if pot_dist < max:
            neighbors.pop(max)
            neighbors[pot_dist] = records[i]

    return next_fvc(patient, neighbors)

# Undoes normalization on FVC data


def unnormal(point):
    return point*diff + min_fvc

# Finds the average of the sequence, better known as the "height"


def avg_seq(patient):
    patient_avg = 0

    for fvc in patient[0]:
        patient_avg += fvc[1]

    patient_avg /= len(patient[0])

    return patient_avg

# calculates the next 3 FVC values given a patient and its neighbors


def next_fvc(patient, neighbors):
    fvc_list = [0, 0, 0]

    times = [fvc[0] for fvc in patient[0][-3:]]

    for key in neighbors:
        for i, _ in enumerate(fvc_list):
            for time in times:
                close = closest([pat[1] for pat in neighbors[key][0]], time)
                fvc_list[i] += neighbors[key][0][close][1]

    return [i / len(neighbors) for i in fvc_list]


def main():
    raw_records = []
    records = []
    row_length = 11

    # Read in the file with type conversions
    with open('./train_preprocessedv2.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        for row in reader:
            raw_records.append([row[1]] + [float(i)
                                           for i in row[2:row_length]])

    
    # Process the time series data
    current_patient = raw_records[0][0]
    patient_data = raw_records[0][4:row_length-1]
    time_record = []

    for record in raw_records:
        if current_patient == record[0]:
            time_record.append((record[1], record[2], record[3]))
        elif current_patient != record[0]:
            records.append([time_record] + patient_data)

            current_patient = record[0]
            patient_data = record[4:row_length-1]

            time_record = []
            time_record.append((record[1], record[2], record[3]))

    records.append([time_record] + patient_data)
    
 

    # Perform k-fold cross-validation
    kf = KFold(n_splits=5)

    rmse_list = []

    k_max = 20

    for k in range(k_max):
        rmse = 0

        for train_index, test_index in kf.split(records):
            training = [records[i] for i in train_index]
            testing = [records[i] for i in test_index]

            training = [patient for patient in training if len(patient[0]) > 3]
            testing = [patient for patient in testing if len(patient[0]) > 3]

            actual = []

            # Take the real last 3 weeks FVC values
            for patient in testing:
                actual.append(unnormal(patient[0][-1][1]))
                actual.append(unnormal(patient[0][-2][1]))
                actual.append(unnormal(patient[0][-3][1]))

            # Perform predictions
            pre_predict = [knn(patient, training, k+1) for patient in testing]
            predict = [item for sublist in pre_predict for item in sublist]

            # Calculate error
            rmse += math.sqrt(mean_squared_error(actual, predict))

        rmse /= kf.get_n_splits()
        
        rmse_list.append(rmse)

    # Process for finding the best k for knn
    min_rmse = rmse_list[0]
    min_index = 0

    for i in range(1, k_max):
        if rmse_list[i] < min_rmse:
            min_index = i
            min_rmse = rmse_list[i]

    print(rmse_list[min_index])
    print(min_index)

    _, ax = plt.subplots()

    plt.plot(range(1, 21), rmse_list, 'ro')

    plt.xlabel('K parameter')
    ax.set_xticks(range(1, 21))
    plt.ylabel('RMSE')

    plt.title('K Parameter Optimization from 5-fold Cross Validation')
    plt.show()

if __name__ == "__main__":
    main()

