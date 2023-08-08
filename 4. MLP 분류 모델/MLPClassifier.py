import numpy as np
from sklearn.neural_network import MLPClassifier
import datetime


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


def MLP_classifier(file, is_week):
    data = np.loadtxt(file, delimiter=',', skiprows=1)

    load_size = 6

    train_data_size = int(np.size(data, 0) * 0.7)  # 학습 데이터:테스트 데이터 = 7:3

    x_train = data[0:train_data_size, 3+load_size:]  # 학습 데이터
    y_train = data[0:train_data_size, 3:3+load_size]  # 학습 데이터에 대한 정답

    x_test = data[train_data_size:, 3+load_size:]  # 테스트 데이터(예측하려고 한 데이터)

    model = MLPClassifier(activation='tanh', solver='adam', alpha=1e-5, hidden_layer_sizes=(load_size*20, 3), random_state=1, max_iter=30000)  # 학습 모델(alpha: 일부러 만드는 오차 정도, random_state: 난수 시드)
    model.fit(X=x_train, y=y_train)  # 학습

    test_predict = model.predict(X=x_test)  # 예측

    result = np.zeros((np.size(test_predict, 0), np.size(test_predict, 1)+3))  # +3은 날짜 입력을 위한 공간
    result[:, 0:3] = data[train_data_size:, 0:3]  # 날짜 입력
    result[:, 3:] = test_predict[:, :]  # 예측 값 입력

    if is_week:
        np.savetxt('MLPClass-Prediction.csv', expand_prediction_result(result), delimiter=',', header='year, month, day', fmt='%4d')
    else:
        np.savetxt('MLPClass-Prediction-5D.csv', expand_prediction_result(result), delimiter=',', header='year, month, day', fmt='%4d')
