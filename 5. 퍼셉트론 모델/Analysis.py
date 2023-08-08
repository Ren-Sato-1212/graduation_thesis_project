import numpy as np
import datetime


def get_index(predict_data_date, answer_date):
    for i in range(0, np.size(answer_date, 0)):
        if int(answer_date[i][0]) == predict_data_date.year and int(answer_date[i][1]) == predict_data_date.month and int(answer_date[i][2]) == predict_data_date.day:
            return i

    return -1


def is_winter_season(month):
    if month >= 11 or month <= 2:
        return True
    else:
        return False


def get_max_time_list(date):
    if is_winter_season(date.month):
        return [13, 14, 20, 21, 22, 25]
    else:
        return [13, 14, 16, 17, 18, 19]


def analysis(predict_file, answer_file, output_file):
    predict_data = np.loadtxt(predict_file, delimiter=',', skiprows=1)

    answer_data = np.loadtxt(answer_file, delimiter=',', skiprows=1)

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for i in range(0, np.size(predict_data, 0)):
        date = datetime.date(int(predict_data[i][0]), int(predict_data[i][1]), int(predict_data[i][2]))

        index = get_index(date, answer_data)

        if index != -1:
            max_time_list = get_max_time_list(date)
            for j in max_time_list:
                if predict_data[i][j] == answer_data[index][j]:
                    if predict_data[i][j] == 1:
                        true_positive = true_positive + 1
                    else:
                        true_negative = true_negative + 1
                else:
                    if predict_data[i][j] == 1:
                        false_positive = false_positive + 1
                    else:
                        false_negative = false_negative + 1

    precision = true_positive / (true_positive + false_positive) * 100  # 정밀도
    recall = true_positive / (true_positive + false_negative) * 100  # 재현율
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative) * 100  # 정확도

    output_file = open(output_file, 'w')
    output_file.write("정밀도: %5.2f%%\n" % precision)
    output_file.write("재현율: %5.2f%%\n" % recall)
    output_file.write("정확도: %5.2f%%\n" % accuracy)

    output_file.close()
