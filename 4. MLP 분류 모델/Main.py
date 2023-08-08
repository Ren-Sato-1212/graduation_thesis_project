import MLPClassifier
import Analysis

MLPClassifier.MLP_classifier('MLInput.csv', True)  # 예측하고자 하는 날로부터 최근 5주간 데이터를 사용
Analysis.analysis('MLPClass-Prediction.csv', 'CSI-MaxLoad-Result-80.csv', True)

MLPClassifier.MLP_classifier('MLContInput.csv', False) # 예측하고자 하는 날로부터 최근 5일간 데이터를 사용
Analysis.analysis('MLPClass-Prediction-5D.csv', 'CSI-MaxLoad-Result-80.csv', False)
