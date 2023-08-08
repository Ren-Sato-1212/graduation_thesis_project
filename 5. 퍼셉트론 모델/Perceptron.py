import numpy as np
import tensorflow as tf
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


def single_layer_perceptron(input_file, is_week):
    input_data = np.loadtxt(input_file, delimiter=',', skiprows=1)

    load_size = 6
    window_size = 5

    train_data_size = int(np.size(input_data, 0) * 0.7)  # 학습 데이터:테스트 데이터 = 7:3

    x_train = input_data[0:train_data_size, 3 + load_size:]  # 학습 데이터
    y_train = input_data[0:train_data_size, 3:3 + load_size]  # 학습 데이터에 대한 정답

    x_test = input_data[train_data_size:, 3 + load_size:]  # 테스트 데이터
    y_test = input_data[train_data_size:, 3:3 + load_size]  # 테스트 데이터에 대한 정답

    INPUT_SIZE = load_size * window_size  # 5일분 데이터 크기
    HIDDEN_SIZE = 2  # 은닉층 노드 개수
    OUTPUT_SIZE = load_size  # 예측 결과 크기

    tf.compat.v1.disable_eager_execution()

    inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, INPUT_SIZE], name='input')
    outputs = tf.compat.v1.placeholder(tf.float32, shape=[None, OUTPUT_SIZE], name='output')

    W = {
        'hidden': tf.Variable(tf.random.normal([INPUT_SIZE, HIDDEN_SIZE])),
        'output': tf.Variable(tf.random.normal([HIDDEN_SIZE, OUTPUT_SIZE]))
    }  # 가중치

    b = {
        'hidden': tf.Variable(tf.random.normal([HIDDEN_SIZE], mean=1.0)),
        'output': tf.Variable(tf.random.normal([OUTPUT_SIZE], mean=1.0))
    }  # 보정치

    hidden = tf.matmul(inputs, W['hidden']) + b['hidden']  # 은닉층
    predict = tf.matmul(hidden, W['output']) + b['output']  # 출력층 -> 활성함수를 사용하지 않음

    error = tf.reduce_mean(tf.square(predict - outputs))  # 오차 함수
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.001)  # 학습률
    train = optimizer.minimize(error)

    EPHOCH = 2000

    session = tf.compat.v1.Session()

    session.run(tf.compat.v1.global_variables_initializer())
    feed = {inputs: x_train, outputs: y_train}
    for epoch in range(EPHOCH):
        session.run(train, feed)

        if epoch % 20 == 0:
            print(epoch, session.run(error, feed))  # 학습

    feed = {inputs: x_test, outputs: y_test}

    predict_value = session.run(predict, feed)

    predict_value[predict_value[:, :] < 0.5] = 0
    predict_value[predict_value[:, :] >= 0.5] = 1  # 예측

    result = np.zeros((np.size(predict_value, 0), np.size(predict_value, 1) + 3))
    result[:, 0:3] = input_data[train_data_size:, 0:3]
    result[:, 3:] = predict_value[:, :]

    if is_week:
        np.savetxt('Perceptron-SingleLayer-Prediction-80.csv', expand_prediction_result(result), delimiter=',', header='year, month, day', fmt='%4d')
    else:
        np.savetxt('Perceptron-SingleLayer-5D-Prediction-80.csv', expand_prediction_result(result), delimiter=',', header='year, month, day', fmt='%4d')


def multi_layer_perceptron(input_file, is_week):
    input_data = np.loadtxt(input_file, delimiter=',', skiprows=1)

    load_size = 6
    window_size = 5

    train_data_size = int(np.size(input_data, 0) * 0.7)  # 학습 데이터:테스트 데이터 = 7:3

    x_train = input_data[0:train_data_size, 3 + load_size:]  # 학습 데이터
    y_train = input_data[0:train_data_size, 3:3 + load_size]  # 학습 데이터에 대한 정답

    x_test = input_data[train_data_size:, 3 + load_size:]  # 테스트 데이터
    y_test = input_data[train_data_size:, 3:3 + load_size]  # 테스트 데이터에 대한 정답

    INPUT_SIZE = load_size * window_size  # 5일분 데이터 크기
    OUTPUT_SIZE = load_size  # 예측 결과 크기

    tf.compat.v1.disable_eager_execution()

    inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, INPUT_SIZE], name='input')
    outputs = tf.compat.v1.placeholder(tf.float32, shape=[None, OUTPUT_SIZE], name='output')

    W = {
        'hidden_1': tf.Variable(tf.random.normal([INPUT_SIZE, INPUT_SIZE*2])),
        'hidden_2': tf.Variable(tf.random.normal([INPUT_SIZE*2, INPUT_SIZE])),
        'output': tf.Variable(tf.random.normal([INPUT_SIZE, OUTPUT_SIZE]))
    }  # 가중치

    b = {
        'hidden_1': tf.Variable(tf.random.normal([INPUT_SIZE*2], mean=1.0)),
        'hidden_2': tf.Variable(tf.random.normal([INPUT_SIZE], mean=1.0)),
        'output': tf.Variable(tf.random.normal([OUTPUT_SIZE], mean=1.0))
    }  # 보정치

    hidden_1 = tf.nn.relu(tf.matmul(inputs, W['hidden_1']) + b['hidden_1'])  # 1층
    hidden_2 = tf.nn.relu(tf.matmul(hidden_1, W['hidden_2']) + b['hidden_2'])  # 2층
    predict = tf.nn.sigmoid(tf.matmul(hidden_2, W['output']) + b['output'])  # 출력층 -> 활성함수 사용

    error = tf.reduce_sum(tf.math.multiply(predict, outputs))  # 오차 함수
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.001)  # 학습률
    train = optimizer.minimize(error)

    EPHOCH = 2000

    session = tf.compat.v1.Session()

    session.run(tf.compat.v1.global_variables_initializer())
    feed = {inputs: x_train, outputs: y_train}
    for epoch in range(EPHOCH):
        session.run(train, feed)

        if epoch % 20 == 0:
            print(epoch, session.run(error, feed))  # 학습

    feed = {inputs: x_test, outputs: y_test}

    predict_value = session.run(predict, feed)

    predict_value[predict_value[:, :] < 0.5] = 0
    predict_value[predict_value[:, :] >= 0.5] = 1  # 예측

    result = np.zeros((np.size(predict_value, 0), np.size(predict_value, 1) + 3))
    result[:, 0:3] = input_data[train_data_size:, 0:3]
    result[:, 3:] = predict_value[:, :]

    if is_week:
        np.savetxt('Perceptron-MultiLayer-Prediction-80.csv', expand_prediction_result(result), delimiter=',', header='year, month, day', fmt='%4d')
    else:
        np.savetxt('Perceptron-MultiLayer-5D-Prediction-80.csv', expand_prediction_result(result), delimiter=',', header='year, month, day', fmt='%4d')


def multiple_perceptron(input_file, is_week):
    input_data = np.loadtxt(input_file, delimiter=',', skiprows=1)

    load_size = 6
    window_size = 5

    train_data_size = int(np.size(input_data, 0) * 0.7)  # 학습 데이터:테스트 데이터 = 7:3

    x_train = input_data[0:train_data_size, 3 + load_size:]  # 학습 데이터

    # 학습 데이터에 대한 정답
    y_train1 = input_data[0:train_data_size, 3:4]
    y_train2 = input_data[0:train_data_size, 4:5]
    y_train3 = input_data[0:train_data_size, 5:6]
    y_train4 = input_data[0:train_data_size, 6:7]
    y_train5 = input_data[0:train_data_size, 7:8]
    y_train6 = input_data[0:train_data_size, 8:9]

    x_test = input_data[train_data_size:, 3 + load_size:]  # 테스트 데이터

    # 테스트 데이터에 대한 정답
    y_test1 = input_data[train_data_size:, 3:4]
    y_test2 = input_data[train_data_size:, 4:5]
    y_test3 = input_data[train_data_size:, 5:6]
    y_test4 = input_data[train_data_size:, 6:7]
    y_test5 = input_data[train_data_size:, 7:8]
    y_test6 = input_data[train_data_size:, 8:9]

    INPUT_SIZE = load_size * window_size  # 5일분 데이터 크기

    tf.compat.v1.disable_eager_execution()

    inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, INPUT_SIZE], name='input')

    output1 = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='output1')
    output2 = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='output2')
    output3 = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='output3')
    output4 = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='output4')
    output5 = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='output5')
    output6 = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='output6')

    # 가중치
    W1 = tf.Variable(tf.random.normal([INPUT_SIZE, 1]))
    W2 = tf.Variable(tf.random.normal([INPUT_SIZE, 1]))
    W3 = tf.Variable(tf.random.normal([INPUT_SIZE, 1]))
    W4 = tf.Variable(tf.random.normal([INPUT_SIZE, 1]))
    W5 = tf.Variable(tf.random.normal([INPUT_SIZE, 1]))
    W6 = tf.Variable(tf.random.normal([INPUT_SIZE, 1]))

    # 보정치
    b1 = tf.Variable(tf.random.normal([1], mean=1.0))
    b2 = tf.Variable(tf.random.normal([1], mean=1.0))
    b3 = tf.Variable(tf.random.normal([1], mean=1.0))
    b4 = tf.Variable(tf.random.normal([1], mean=1.0))
    b5 = tf.Variable(tf.random.normal([1], mean=1.0))
    b6 = tf.Variable(tf.random.normal([1], mean=1.0))

    # 출력층 -> 활성함수 사용
    predict1 = tf.sigmoid(tf.matmul(inputs, W1) + b1)
    predict2 = tf.sigmoid(tf.matmul(inputs, W2) + b2)
    predict3 = tf.sigmoid(tf.matmul(inputs, W3) + b3)
    predict4 = tf.sigmoid(tf.matmul(inputs, W4) + b4)
    predict5 = tf.sigmoid(tf.matmul(inputs, W5) + b5)
    predict6 = tf.sigmoid(tf.matmul(inputs, W6) + b6)

    # 오차함수
    error1 = tf.reduce_sum(output1 * tf.math.log(predict1 + 0.0000001) + (1 - output1) * tf.math.log(1 - predict1 - 0.0000001), axis=1)
    error2 = tf.reduce_sum(output2 * tf.math.log(predict2 + 0.0000001) + (1 - output2) * tf.math.log(1 - predict2 - 0.0000001), axis=1)
    error3 = tf.reduce_sum(output3 * tf.math.log(predict3 + 0.0000001) + (1 - output3) * tf.math.log(1 - predict3 - 0.0000001), axis=1)
    error4 = tf.reduce_sum(output4 * tf.math.log(predict4 + 0.0000001) + (1 - output4) * tf.math.log(1 - predict4 - 0.0000001), axis=1)
    error5 = tf.reduce_sum(output5 * tf.math.log(predict5 + 0.0000001) + (1 - output5) * tf.math.log(1 - predict5 - 0.0000001), axis=1)
    error6 = tf.reduce_sum(output6 * tf.math.log(predict6 + 0.0000001) + (1 - output6) * tf.math.log(1 - predict6 - 0.0000001), axis=1)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.001)  # 학습률

    train1 = optimizer.minimize(error1)
    train2 = optimizer.minimize(error2)
    train3 = optimizer.minimize(error3)
    train4 = optimizer.minimize(error4)
    train5 = optimizer.minimize(error5)
    train6 = optimizer.minimize(error6)

    EPHOCH = 2000

    session = tf.compat.v1.Session()

    session.run(tf.compat.v1.global_variables_initializer())

    feed = {inputs: x_train, output1: y_train1}
    for epoch in range(EPHOCH):
        session.run(train1, feed)

        if epoch % 200 == 0:
            print(epoch, session.run(error1, feed))  # 학습

    feed = {inputs: x_test, output1: y_test1}

    predict_value1 = session.run(predict1, feed)

    predict_value1[predict_value1[:, :] < 0.5] = 0
    predict_value1[predict_value1[:, :] >= 0.5] = 1  # 예측

    feed = {inputs: x_train, output2: y_train2}
    for epoch in range(EPHOCH):
        session.run(train2, feed)

        if epoch % 200 == 0:
            print(epoch, session.run(error2, feed))  # 학습

    feed = {inputs: x_test, output2: y_test2}

    predict_value2 = session.run(predict2, feed)

    predict_value2[predict_value2[:, :] < 0.5] = 0
    predict_value2[predict_value2[:, :] >= 0.5] = 1  # 예측

    feed = {inputs: x_train, output3: y_train3}
    for epoch in range(EPHOCH):
        session.run(train3, feed)

        if epoch % 200 == 0:
            print(epoch, session.run(error3, feed))  # 학습

    feed = {inputs: x_test, output3: y_test3}

    predict_value3 = session.run(predict3, feed)

    predict_value3[predict_value3[:, :] < 0.5] = 0
    predict_value3[predict_value3[:, :] >= 0.5] = 1  # 예측

    feed = {inputs: x_train, output4: y_train4}
    for epoch in range(EPHOCH):
        session.run(train4, feed)

        if epoch % 200 == 0:
            print(epoch, session.run(error4, feed))  # 학습

    feed = {inputs: x_test, output4: y_test4}

    predict_value4 = session.run(predict4, feed)

    predict_value4[predict_value4[:, :] < 0.5] = 0
    predict_value4[predict_value4[:, :] >= 0.5] = 1  # 예측

    feed = {inputs: x_train, output5: y_train5}
    for epoch in range(EPHOCH):
        session.run(train5, feed)

        if epoch % 200 == 0:
            print(epoch, session.run(error5, feed))  # 학습

    feed = {inputs: x_test, output5: y_test5}

    predict_value5 = session.run(predict5, feed)

    predict_value5[predict_value5[:, :] < 0.5] = 0
    predict_value5[predict_value5[:, :] >= 0.5] = 1  # 예측

    feed = {inputs: x_train, output6: y_train6}
    for epoch in range(EPHOCH):
        session.run(train6, feed)

        if epoch % 200 == 0:
            print(epoch, session.run(error6, feed))  # 학습

    feed = {inputs: x_test, output6: y_test6}

    predict_value6 = session.run(predict6, feed)

    predict_value6[predict_value6[:, :] < 0.5] = 0
    predict_value6[predict_value6[:, :] >= 0.5] = 1  # 예측

    result = np.zeros((np.size(predict_value1, 0), 6 + 3))
    result[:, 0:3] = input_data[train_data_size:, 0:3]
    result[:, 3:4] = predict_value1[:, :]
    result[:, 4:5] = predict_value2[:, :]
    result[:, 5:6] = predict_value3[:, :]
    result[:, 6:7] = predict_value4[:, :]
    result[:, 7:8] = predict_value5[:, :]
    result[:, 8:9] = predict_value6[:, :]

    if is_week:
        np.savetxt('Perceptron-MultipleModel-Prediction-80.csv', expand_prediction_result(result), delimiter=',', header='year, month, day', fmt='%4d')
    else:
        np.savetxt('Perceptron-MultipleModel-5D-Prediction-80.csv', expand_prediction_result(result), delimiter=',', header='year, month, day', fmt='%4d')
