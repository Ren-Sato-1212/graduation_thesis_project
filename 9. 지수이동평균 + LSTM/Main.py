import LSTM
import ExponentialMovingAverage
import Merge
import Analysis

LSTM.LSTM_2layer('MLContFloatInput.csv', 'LSTM-2Layer-5D-80Cutoff-FloatInput-Prediction.csv')
ExponentialMovingAverage.exponential_moving_average('TotalPower.csv')
Merge.merge('LSTM-2Layer-5D-80Cutoff-FloatInput-Prediction.csv', 'EA-Prediction-80.csv')

Analysis.analysis('LSTM-EA-MergeAnd.csv', 'CSI-MaxLoad-Result-80.csv', 'LSTM-EA-MergeAnd-Performance.txt')
Analysis.analysis('LSTM-EA-MergeOr.csv', 'CSI-MaxLoad-Result-80.csv', 'LSTM-EA-MergeOr-Performance.txt')
