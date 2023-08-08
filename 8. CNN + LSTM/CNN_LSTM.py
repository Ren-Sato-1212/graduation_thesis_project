import numpy as np
import datetime
import keras


def is_winter_season(month):
    if month >= 11 or month <= 2:
        return True

    else:
        return False


def get_load_time_interval(date):
    if is_winter_season(date.month):
        return [13, 14, 20, 21, 22, 25]

    else:
        return [13, 14, 16, 17, 18, 19]


def expand_prediction_result(predict_data):
    predict_result = np.zeros((np.size(predict_data, 0), 27))

    for i in range(0, np.size(predict_data, 0)):
        date = datetime.date(int(predict_data[i][0]), int(predict_data[i][1]), int(predict_data[i][2]))

        expand_index = get_load_time_interval(date)

        predict_result[i, 0:3] = predict_data[i, 0:3]
        predict_result[i, expand_index] = predict_data[i, 3:]

    return predict_result


def CNN_LSTM(input_file):
    input_data = np.loadtxt(input_file, delimiter=',', skiprows=1)

    train_data_size = int(np.size(input_data, 0) * 0.7)

    load_size = 6

    x_train = input_data[0:train_data_size, 3+load_size:]
    y_train = input_data[0:train_data_size, 3:3+load_size]

    x_test = input_data[train_data_size:, 3+load_size:]

    x_train = np.reshape(x_train, (train_data_size, np.size(x_train, 1), 1))
    x_test = np.reshape(x_test, (np.size(input_data, 0) - train_data_size, np.size(x_test, 1), 1))

    model = keras.Sequential()

    model.add(keras.layers.Conv1D(264, 6, activation='relu', input_shape=(np.size(x_train, 1), 1)))
    model.add(keras.layers.MaxPool1D(pool_size=6))
    model.add(keras.layers.LSTM(128))
    model.add(keras.layers.Dense(load_size, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=1000)

    test_predict = model.predict(x_test, batch_size=1)

    test_predict[test_predict[:, :] < 0.5] = 0
    test_predict[test_predict[:, :] >= 0.5] = 1

    result = np.zeros((np.size(test_predict, 0), np.size(test_predict, 1) + 3))
    result[:, 0:3] = input_data[train_data_size:, 0:3]
    result[:, 3:] = test_predict[:, :]

    np.savetxt('CNNLSTM-Prediction.csv', expand_prediction_result(result), delimiter=',', header='year, month, day', fmt='%4d')
