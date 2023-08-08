import numpy as np
from workalendar.asia import SouthKorea
import datetime


def create_SI_matrix(simple_moving_average_matrix, previous_day_average_matrix):  # SI 계산
    SI_matrix = np.zeros(simple_moving_average_matrix.shape)

    for i in range(np.size(simple_moving_average_matrix, 0)):
        SI_matrix[i][0:3] = simple_moving_average_matrix[i][0:3]  # 날짜 입력

        SI_matrix[i][3] = simple_moving_average_matrix[i][3] - previous_day_average_matrix[i]
        for j in range(4, 27):
            SI_matrix[i][j] = simple_moving_average_matrix[i][j] - simple_moving_average_matrix[i][j-1]

    return SI_matrix


def create_CSI_matrix(SI_matrix):  # 누적 SI(CSI) 계산
    CSI_matrix = np.zeros(SI_matrix.shape)

    for i in range(np.size(SI_matrix, 0)):
        CSI_matrix[i][0:4] = SI_matrix[i][0:4]

        for j in range(4, 27):
            CSI_matrix[i][j] = SI_matrix[i][j] + CSI_matrix[i][j-1]

    return CSI_matrix


def create_CSI_percent_matrix(CSI_matrix):  # 누적 SI(CSI)의 최대값에 대한 비율 계산
    CSI_percent_matrix = np.zeros(CSI_matrix.shape)

    CSI_max_matrix = np.amax(CSI_matrix[:, 3:], axis=1)

    for i in range(np.size(CSI_matrix, 0)):
        CSI_percent_matrix[i][0:3] = CSI_matrix[i][0:3]

        for j in range(3, 27):
            CSI_percent_matrix[i][j] = CSI_matrix[i][j] / CSI_max_matrix[i] * 100

    return CSI_percent_matrix


def create_CSI_percent_classfication_matrix(CSI_percent_matrix):  # ESS에 저장된 전기 에너지를 사용할 시간대 선택
    CSI_percent_classfication_matrix = np.zeros(CSI_percent_matrix.shape)

    for i in range(0, np.size(CSI_percent_matrix, 0)):
        CSI_percent_classfication_matrix[i, 0:3] = CSI_percent_matrix[i][0:3]

        for j in range(3, 27):
            if CSI_percent_matrix[i][j] >= 80:
                CSI_percent_classfication_matrix[i][j] = 1
            else:
                CSI_percent_classfication_matrix[i][j] = 0

    return CSI_percent_classfication_matrix


def exponential_moving_average(input_file):  # 지수이동평균 모델
    input_data = np.loadtxt(input_file, delimiter=',', skiprows=1)

    previous_days_size = 7 * 2 + 1  # 예측하고자 하는 날로부터 최근 2주간 데이터 사용

    calendar = SouthKorea()
    work_days_size = 0
    for i in range(previous_days_size, np.size(input_data, 0)):
        date = datetime.date(int(input_data[i][0]), int(input_data[i][1]), int(input_data[i][2]))
        if calendar.is_working_day(date):
            work_days_size = work_days_size + 1

    exponential_moving_average_matrix = np.zeros((work_days_size-1, 3+24))
    previous_day_average_matrix = np.zeros(work_days_size-1)

    alpha = 0.5  # 가중치

    index = 0
    for i in range(previous_days_size+1, np.size(input_data, 0)):
        date = datetime.date(int(input_data[i][0]), int(input_data[i][1]), int(input_data[i][2]))
        if calendar.is_working_day(date):
            exponential_moving_average_matrix[index][0:3] = input_data[i][0:3]

            previous_work_days_size = 0
            for k in range(0, previous_days_size):
                date = datetime.date(int(input_data[i-k-1][0]), int(input_data[i-k-1][1]), int(input_data[i-k-1][2]))
                if calendar.is_working_day(date):
                    for j in range(3+0, 3+24):
                        exponential_moving_average_matrix[index][j] = exponential_moving_average_matrix[index][j] + alpha * pow(1-alpha, previous_work_days_size) * input_data[i-k-1][j]  # 예측하고자 하는 날로부터 먼 날의 데이터일수록 전기 사용량을 예측하는데 그 데이터가 덜 반영이 되도록 조치를 취함

                    previous_day_average_matrix[index] = previous_day_average_matrix[index] + alpha * pow(1-alpha, previous_work_days_size) * input_data[i-k-2][3+23]  # 예측하고자 하는 날로부터 먼 날의 데이터일수록 전기 사용량을 예측하는데 그 데이터가 덜 반영이 되도록 조치를 취함

                    previous_work_days_size = previous_work_days_size + 1

            index = index + 1

    SI_matrix = create_SI_matrix(exponential_moving_average_matrix, previous_day_average_matrix)  # SI
    CSI_matrix = create_CSI_matrix(SI_matrix)  # 누적 SI(CSI)
    CSI_percent_matrix = create_CSI_percent_matrix(CSI_matrix)  # 누적 SI(CSI)의 최대값에 대한 비율
    CSI_percent_classification_matrix = create_CSI_percent_classfication_matrix(CSI_percent_matrix)  # ESS에 저장된 전기 에너지를 사용할 시간대 선택

    np.savetxt('EA-Prediction-80.csv', CSI_percent_classification_matrix, delimiter=',', header='year, month, day', fmt='%4d')  # 출력
