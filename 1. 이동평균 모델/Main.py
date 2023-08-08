import SimpleMovingAverage
import Analysis

SimpleMovingAverage.simple_moving_average('TotalPower.csv', 8)  # 예측하고자 하는 날로부터 최근 8주간 데이터를 사용
SimpleMovingAverage.simple_moving_average('TotalPower.csv', 4)  # 예측하고자 하는 날로부터 최근 4주간 데이터를 사용
SimpleMovingAverage.simple_moving_average('TotalPower.csv', 2)  # 예측하고자 하는 날로부터 최근 2주간 데이터를 사용

# 성능 확인
Analysis.analysis('MA-Prediction-80-8w.csv', 'CSI-MaxLoad-Result-80.csv', 8)
Analysis.analysis('MA-Prediction-80-4w.csv', 'CSI-MaxLoad-Result-80.csv', 4)
Analysis.analysis('MA-Prediction-80-2w.csv', 'CSI-MaxLoad-Result-80.csv', 2)
