import datetime
import numpy as np
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


def LSTM_64(learning_file, class_file):
    learning_data = np.loadtxt(learning_file, delimiter=',', skiprows=1)
    class_data = np.loadtxt(class_file, delimiter=',', skiprows=1)

    all_train_size = int(np.size(learning_data, 0) * 0.7)
    value_size = int(all_train_size * 0.2)  # 모델이 학습 중간중간에 학습이 잘 되고 있는지 확인하기 위한 데이터의 갯수
    train_size = all_train_size - value_size  # 학습 데이터 갯수
    test_size = np.size(learning_data, 0) - all_train_size  # 테스트 데이터 갯수

    class_size = 64  # 분류 갯수

    x_train = learning_data[0:train_size, 3 + 6:]  # 학습 데이터
    y_train = keras.utils.to_categorical(class_data[0:train_size, 3], num_classes=class_size)  # 학습 데이터에 대한 정답

    x_value = learning_data[train_size:all_train_size, 3 + 6:]  # 모델이 학습 중간중간에 학습이 잘 되고 있는지 확인하기 위한 데이터
    y_value = keras.utils.to_categorical(class_data[train_size:all_train_size, 3], num_classes=class_size)  # 모델이 학습 중간중간에 학습이 잘 되고 있는지 확인하기 위한 데이터에 대한 정답

    x_test = learning_data[all_train_size:, 3 + 6:]  # 테스트 데이터

    days_size = 5
    per_day_data_size = 6

    x_train = np.reshape(x_train, (train_size, days_size, per_day_data_size))
    x_value = np.reshape(x_value, (value_size, days_size, per_day_data_size))
    x_test = np.reshape(x_test, (test_size, days_size, per_day_data_size))

    model = keras.models.Sequential()

    model.add(keras.layers.LSTM(128, batch_input_shape=(1, days_size, per_day_data_size), stateful=True, return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(128, return_sequences=False))
    model.add(keras.layers.Dense(class_size))
    model.add(keras.layers.Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs_count = 100

    model.fit(x_train, y_train, epochs=epochs_count, batch_size=1, verbose=1, validation_data=(x_value, y_value))  # 학습

    test_predict = model.predict_classes(x_test, batch_size=1)  # 예측

    result = np.zeros((np.size(test_predict, 0), 6 + 3))
    for i in range(np.size(test_predict, 0)):
        value = test_predict[i]

        for j in range(6):
            result[i, 3+j] = value % 2
            value = int(value / 2)

    result[:, 0:3] = learning_data[all_train_size:, 0:3]

    np.savetxt('LSTMClass-Prediction.csv', expand_prediction_result(result), delimiter=',', header='year, month, day', fmt='%4d')


def LSTM_6(learning_file, class_file):
    learning_data = np.loadtxt(learning_file, delimiter=',', skiprows=1)
    class_data = np.loadtxt(class_file, delimiter=',', skiprows=1)

    all_train_size = int(np.size(learning_data, 0) * 0.7)
    value_size = int(all_train_size * 0.2)  # 모델이 학습 중간중간에 학습이 잘 되고 있는지 확인하기 위한 데이터의 갯수
    train_size = all_train_size - value_size  # 학습 데이터 갯수
    test_size = np.size(learning_data, 0) - all_train_size  # 테스트 데이터 갯수

    x_train = learning_data[0:train_size, 3 + 6:]  # 학습 데이터
    y_train = class_data[0:train_size, 3:]  # 학습 데이터에 대한 정답

    x_value = learning_data[train_size:all_train_size, 3 + 6:]  # 모델이 학습 중간중간에 학습이 잘 되고 있는지 확인하기 위한 데이터
    y_value = class_data[train_size:all_train_size, 3:]  # 모델이 학습 중간중간에 학습이 잘 되고 있는지 확인하기 위한 데이터에 대한 정답

    x_test = learning_data[all_train_size:, 3 + 6:]  # 테스트 데이터

    days_size = 5
    per_day_data_size = 6

    x_train = np.reshape(x_train, (train_size, days_size, per_day_data_size))
    x_value = np.reshape(x_value, (value_size, days_size, per_day_data_size))
    x_test = np.reshape(x_test, (test_size, days_size, per_day_data_size))

    model = keras.models.Sequential()

    model.add(keras.layers.LSTM(128, batch_input_shape=(1, days_size, per_day_data_size), stateful=True, return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(128, return_sequences=False))
    model.add(keras.layers.Dense(per_day_data_size))
    model.add(keras.layers.Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs_count = 200

    model.fit(x_train, y_train, epochs=epochs_count, batch_size=1, verbose=1, validation_data=(x_value, y_value))  # 학습

    test_predict = model.predict(x_test, batch_size=1)  # 예측

    test_predict[test_predict[:, :] < 0.5] = 0
    test_predict[test_predict[:, :] >= 0.5] = 1

    result = np.zeros((np.size(test_predict, 0), np.size(test_predict, 1) + 3))
    result[:, 0:3] = learning_data[all_train_size:, 0:3]
    result[:, 3:] = test_predict[:, :]

    np.savetxt('LSTMClassBinary-Prediction.csv', expand_prediction_result(result), delimiter=',', header='year, month, day', fmt='%4d')
