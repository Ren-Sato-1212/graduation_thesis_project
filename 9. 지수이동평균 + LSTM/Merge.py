import numpy as np

def merge(file1, file2):
    data1 = np.loadtxt(file1, delimiter=',', skiprows=1)
    data2 = np.loadtxt(file2, delimiter=',', skiprows=1)

    if data1.shape[0] <= data2.shape[0]:
        small_data = data1
        large_data = data2

    else:
        small_data = data2
        large_data = data1

    found_index = -1
    for i in range(0, large_data.shape[0]):
        if large_data[i, 0] == small_data[0, 0] and large_data[i, 1] == small_data[0, 1] and large_data[i, 2] == small_data[0, 2]:
            found_index = i
            break

    merge_and_output = np.zeros((np.size(small_data, 0), np.size(small_data, 1)))
    merge_or_output = np.zeros((np.size(small_data, 0), np.size(small_data, 1)))

    for i in range(0, small_data.shape[0]):
        merge_and_output[i, 0:3] = small_data[i, 0:3]
        merge_or_output[i, 0:3] = small_data[i, 0:3]

        for j in range(3, np.size(small_data, 1)):
            merge_and_output[i, j] = int(small_data[i, j] > 0 and large_data[i+found_index, j] > 0)
            merge_or_output[i, j] = int(small_data[i, j] > 0 or large_data[i+found_index, j] > 0)

    np.savetxt('LSTM-EA-MergeAnd.csv', merge_and_output, delimiter=',', header='year, month, day', fmt='%4d')
    np.savetxt('LSTM-EA-MergeOr.csv', merge_or_output, delimiter=',', header='year, month, day', fmt='%4d')
