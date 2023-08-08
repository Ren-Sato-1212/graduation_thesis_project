import ExponentialMovingAverage
import Analysis

ExponentialMovingAverage.exponential_moving_average('TotalPower.csv')
Analysis.analysis('EA-Prediction-80.csv', 'CSI-MaxLoad-Result-80.csv')
