import KDN
import Analysis

KDN.KDN_5w('CSI-Raw.csv', 'CSI-MaxLoad-80-Cutoff-Normalized-Binary-Sliced.csv')  # 예측하고자 하는 날로부터 최근 5주간 데이터를 사용
Analysis.analysis('KDN-MaxLoad-Prediction-Binary-80.csv', 'CSI-MaxLoad-Result-80.csv', True)

KDN.KDN_5d('CSI-Raw.csv', 'CSI-MaxLoad-80-Cutoff-Normalized-Binary-Sliced.csv')  # 예측하고자 하는 날로부터 최근 5일간 데이터를 사용
Analysis.analysis('KDN5D-MaxLoad-Prediction-Binary-80.csv', 'CSI-MaxLoad-Result-80.csv', False)
