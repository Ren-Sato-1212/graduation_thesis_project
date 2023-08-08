import CNN_LSTM
import Analysis

CNN_LSTM.CNN_LSTM('MLContInput.csv')
Analysis.analysis('CNNLSTM-Prediction.csv', 'CSI-MaxLoad-Result-80.csv', 'CNNLSTM-Performance.txt')
