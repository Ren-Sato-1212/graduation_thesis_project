import LSTMDeformation
import Analysis

LSTMDeformation.LSTM_2layer('MLContInput.csv', 'LSTM-2Layer-5D-Prediction.csv')
LSTMDeformation.LSTM_2layer('MLContFloatInput.csv', 'LSTM-2Layer-5D-80Cutoff-FloatInput-Prediction.csv')
LSTMDeformation.LSTM_2layer('CSIContFloatInput.csv', 'LSTM-2Layer-5D-FloatInput-Prediction.csv')
Analysis.analysis('LSTM-2Layer-5D-Prediction.csv', 'CSI-MaxLoad-Result-80.csv', 'LSTM-2Layer-5D-Performance.txt')
Analysis.analysis('LSTM-2Layer-5D-80Cutoff-FloatInput-Prediction.csv', 'CSI-MaxLoad-Result-80.csv', 'LSTM-2Layer-5D-80Cutoff-FloatInput-Performance.txt')
Analysis.analysis('LSTM-2Layer-5D-FloatInput-Prediction.csv', 'CSI-MaxLoad-Result-80.csv', 'LSTM-2Layer-5D-FloatInput-Performance.txt')

LSTMDeformation.LSTM_3layer('MLContInput.csv', 'LSTM-3Layer-5D-Prediction.csv')
Analysis.analysis('LSTM-3Layer-5D-Prediction.csv', 'CSI-MaxLoad-Result-80.csv', 'LSTM-3Layer-5D-Performance.txt')
