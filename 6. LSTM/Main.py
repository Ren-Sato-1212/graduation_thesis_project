import LSTM
import Analysis

LSTM.LSTM_64('CSIContFloatInput.csv', 'MLContClassInput.csv')  # 64개 클래스로 예측하는 LSTM
LSTM.LSTM_6('CSIContFloatInput.csv', 'MLContClassBinaryInput.csv')  # 6개 클래스로 예측하는 LSTM

Analysis.analysis('LSTMClass-Prediction.csv', 'CSI-MaxLoad-Result-80.csv', 'LSTMClass-Performance.txt')
Analysis.analysis('LSTMClassBinary-Prediction.csv', 'CSI-MaxLoad-Result-80.csv', 'LSTMClassBinary-Performance.txt')
