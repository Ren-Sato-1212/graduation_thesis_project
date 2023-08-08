import numpy as np
from workalendar.asia import SouthKorea
import datetime
from sklearn.linear_model import Perceptron


def get_K_fold_index(work_data):  # 교차검증
    K_fold_result = np.zeros(np.size(work_data, 0))
    max_learning_count = 200  # 학습횟수

    for i in range(0, np.size(work_data, 0)):
        learning_data = np.copy(work_data)

        for j in range(0, np.size(work_data, 1)):
            learning_data[i][j] = 0.

        x_train = learning_data[:, 3:]  # 해당 행을 제외하고, 원본 그대로(해당 행은 테스트 데이터로 사용)
        y_train = np.zeros(np.size(x_train, 0))  # 각 행의 '1'의 개수를 나타냄

        is_nonzero = False

        for j in range(0, np.size(x_train, 0)):
            for k in range(0, np.size(x_train, 1)):
                if x_train[j][k] >= 1:
                    y_train[j] = y_train[j] + 1
                    is_nonzero = True

        if is_nonzero:
            model = Perceptron(max_iter=max_learning_count, tol=1e-3).fit(x_train, y_train)  # 퍼셉트론 모델(은닉층이 없는 신경망)
            x_test = work_data[i, 3:]  # 테스트 데이터
            x_test = x_test.reshape((1, np.size(x_test)))
            y_test_predict = model.predict(x_test)  # 테스트 데이터 예측

            y_test = 0  # 테스트 데이터에 대한 정답
            for k in range(0, np.size(x_test, 1)):
                if x_test[0][k] >= 1:
                    y_test = y_test + 1

            if abs(y_test_predict - y_test) < 0.001:  # 예측과 정답이 같은지 확인
                K_fold_result[i] = 1
            else:
                K_fold_result[i] = 0

        else:
            K_fold_result[i] = 1

    return K_fold_result


def load_array_compare(array_A, array_B):  # 예측 데이터끼리 비교
    for i in range(0, np.size(array_A)):
        if array_A[i] >= 1 and array_B[i] < 1:
            return False

        elif array_A[i] < 1 and array_B[i] >= 1:
            return False

    return True


def get_occur_index(work_data, work_data_index, K_fold_index):  # 예측 데이터 결정(1차)
    nonzero_count = np.count_nonzero(K_fold_index)  # 0이 아닌 수의 개수를 셈

    if nonzero_count == 0:  # 예측 데이터 없음
        return -1, work_data_index

    elif nonzero_count == 1:  # 예측 데이터가 1개만
        nonzero_array = np.flatnonzero(K_fold_index)  # k_fold_index 배열에서 0이 아닌 수의 위치 확인
        test_index = nonzero_array[0]

        return work_data_index[test_index], []

    else:  # 예측 데이터가 2개 이상
        nonzero_array = np.flatnonzero(K_fold_index)
        test_index = nonzero_array[0]

        for i in nonzero_array:
            if load_array_compare(work_data[test_index, 3:], work_data[i, 3:]) == False:  # 예측 데이터끼리 비교했을 때, 일치하지 않은 경우
                return -1, work_data_index[nonzero_array]

        return work_data_index[test_index], []  # 예측 데이터끼리 비교했을 때, 일치하는 경우


def find_candidate_index(index, candiate_array):
    for i in range(0, np.size(candiate_array)):
        if candiate_array[i] == index:
            return i

    return -1


def get_number_of_similar_items(rand_sample_data, selected_index, cutoff):
    cofactor = np.zeros((np.size(rand_sample_data, 0)))

    for i in range(0, np.size(rand_sample_data, 0)):
        corval = np.corrcoef(rand_sample_data[selected_index, 3:], rand_sample_data[i, 3:])
        cofactor[i] = corval[0][1]

    cofactor = np.where(cofactor[:] >= cutoff)

    return np.count_nonzero(cofactor)


def get_random_sampling(CSI_data, i, candiate_array, CSI_cutoff_data):  # 예측 데이터 결정(2차)
    day_block_size = 56  # 전체 데이터 구간의 길이를 정함

    calendar = SouthKorea()

    data = np.zeros((day_block_size, np.size(CSI_data, 1)))
    index = 0
    candidate_index = np.zeros((np.size(candiate_array)), dtype=int)

    for j in range(0, day_block_size):
        date = datetime.date(int(CSI_data[i-j-1][0]), int(CSI_data[i-j-1][1]), int(CSI_data[i-j-1][2]))

        if calendar.is_working_day(date):
            data[index] = CSI_data[i-j-1]

            candidate = find_candidate_index(i-j-1, candiate_array)
            if candidate != -1:
                candidate_index[candidate] = index

            index = index + 1

    rand_sample_data = data[:index, :]  # 전체 데이터

    cofactor_save = np.zeros((np.size(candidate_index, 0)))
    for i in range(0, np.size(candidate_index, 0)):
        cofactor_save[i] = get_number_of_similar_items(rand_sample_data, candidate_index[i], 0.8)  # 예측된 데이터별로 전체 데이터와의 상관관계가 80%이상인 것에 대한 갯수 구하기

    chosen_index = np.argmax(cofactor_save)

    return CSI_cutoff_data[int(candiate_array[chosen_index])][3:]  # 전체 데이터와의 상관관계가 80%이상인 것에 대한 갯수가 가장 많은 것을 최종 예측 데이터로 정함


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


def KDN_5w(CSI_file, CSI_cutoff_file):
    CSI_data = np.loadtxt(CSI_file, delimiter=',', skiprows=1)
    CSI_cutoff_data = np.loadtxt(CSI_cutoff_file, delimiter=',', skiprows=1)

    previous_days_size = 7 * 8  # 8주간 데이터(전체 데이터)
    window_size = 5  # 5주간 데이터(실제 사용할 데이터)

    work_data = np.zeros((window_size, np.size(CSI_cutoff_data, 1)))  # 해당일과 같은 요일의 데이터
    work_data_index = np.zeros(window_size)  # work_data 원본 파일 위치

    calendar = SouthKorea()

    predict_data = np.zeros((np.size(CSI_cutoff_data, 0) - previous_days_size, np.size(CSI_cutoff_data, 1)))  # 예측 데이터
    predict_index = 0

    for i in range(previous_days_size, np.size(CSI_cutoff_data, 0)):
        predict_date = datetime.date(int(CSI_cutoff_data[i][0]), int(CSI_cutoff_data[i][1]), int(CSI_cutoff_data[i][2]))  # 예측하고자 하는 날

        if calendar.is_working_day(predict_date):  # 해당일이 근무일인지 아닌지 확인
            previous_work_days_size = 0

            for k in range(0, previous_days_size):
                previous_date = datetime.date(int(CSI_cutoff_data[i-k-1][0]), int(CSI_cutoff_data[i-k-1][1]), int(CSI_cutoff_data[i-k-1][2]))  # 예측하고자 하는 날의 이전일

                if predict_date.weekday() == previous_date.weekday() and calendar.is_working_day(previous_date):  # 예측하고자 하는 날의 이전일이 예측하고자 하는 날의 요일과 같은지 그리고 근무일인지 아닌지 확인
                    work_data[previous_work_days_size] = CSI_cutoff_data[i-k-1]
                    work_data_index[previous_work_days_size] = i-k-1

                    previous_work_days_size = previous_work_days_size + 1

                    if previous_work_days_size == window_size:  # 예측하고자 하는 날로부터 최근 5주간 데이터만 사용하기 위한 조치
                        break

            K_fold_index = get_K_fold_index(work_data)  # 교차검증

            candidate, candiate_array = get_occur_index(work_data, work_data_index, K_fold_index)  # 예측 데이터 결정(1차)

            predict_data[predict_index][0:3] = CSI_cutoff_data[i][0:3]  # 날짜 입력

            if candidate != -1:  # 앞서서 해당 예측 데이터가 정해진 경우
                predict_data[predict_index][3:] = CSI_cutoff_data[int(candidate)][3:]

            else:  # 예측 데이터를 정할 필요가 있는 경우
                predict_data[predict_index][3:] = get_random_sampling(CSI_data, i, candiate_array, CSI_cutoff_data)  # 예측 데이터 결정(2차)

            predict_index = predict_index + 1

    predict_data = predict_data[0:predict_index, ]

    # 출력
    np.savetxt('KDN-MaxLoad-Prediction-Binary-80.csv', expand_prediction_result(predict_data), delimiter=',', header='year, month, day', fmt='%4d')

def KDN_5d(CSI_file, CSI_cutoff_file):
    CSI_data = np.loadtxt(CSI_file, delimiter=',', skiprows=1)
    CSI_cutoff_data = np.loadtxt(CSI_cutoff_file, delimiter=',', skiprows=1)

    previous_days_size = 7 * 2 + 1  # 2주간 데이터(전체 데이터)
    window_size = 5  # 5일간 데이터(실제 사용할 데이터)

    work_data = np.zeros((window_size, np.size(CSI_cutoff_data, 1)))  # 해당일과 같은 요일의 데이터
    work_data_index = np.zeros(window_size)  # work_data 원본 파일 위치

    calendar = SouthKorea()

    predict_data = np.zeros((np.size(CSI_cutoff_data, 0) - previous_days_size, np.size(CSI_cutoff_data, 1)))  # 예측 데이터
    predict_index = 0

    for i in range(previous_days_size, np.size(CSI_cutoff_data, 0)):
        predict_date = datetime.date(int(CSI_cutoff_data[i][0]), int(CSI_cutoff_data[i][1]), int(CSI_cutoff_data[i][2]))  # 예측하고자 하는 날

        if calendar.is_working_day(predict_date):  # 해당일이 근무일인지 아닌지 확인
            previous_work_days_size = 0

            for k in range(0, previous_days_size):
                previous_date = datetime.date(int(CSI_cutoff_data[i-k-1][0]), int(CSI_cutoff_data[i-k-1][1]), int(CSI_cutoff_data[i-k-1][2]))  # 예측하고자 하는 날의 이전일

                if calendar.is_working_day(previous_date):  # 예측하고자 하는 날의 이전일이 근무일인지 아닌지 확인
                    work_data[previous_work_days_size] = CSI_cutoff_data[i-k-1]
                    work_data_index[previous_work_days_size] = i-k-1

                    previous_work_days_size = previous_work_days_size + 1

                    if previous_work_days_size == window_size:  # 예측하고자 하는 날로부터 최근 5일간 데이터만 사용하기 위한 조치
                        break

            K_fold_index = get_K_fold_index(work_data)  # 교차검증

            candidate, candiate_array = get_occur_index(work_data, work_data_index, K_fold_index)  # 예측 데이터 결정(1차)

            predict_data[predict_index][0:3] = CSI_cutoff_data[i][0:3]  # 날짜 입력

            if candidate != -1:  # 앞서서 해당 예측 데이터가 정해진 경우
                predict_data[predict_index][3:] = CSI_cutoff_data[int(candidate)][3:]

            else:  # 예측 데이터를 정할 필요가 있는 경우
                predict_data[predict_index][3:] = get_random_sampling(CSI_data, i, candiate_array, CSI_cutoff_data)  # 예측 데이터 결정(2차)

            predict_index = predict_index + 1

    predict_data = predict_data[0:predict_index, ]

    # 출력
    np.savetxt('KDN5D-MaxLoad-Prediction-Binary-80.csv', expand_prediction_result(predict_data), delimiter=',', header='year, month, day', fmt='%4d')
