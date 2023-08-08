import Perceptron
import Analysis

Perceptron.single_layer_perceptron('MLInput.csv', True)
Perceptron.multi_layer_perceptron('MLInput.csv', True)
Perceptron.multiple_perceptron('MLInput.csv', True)

Perceptron.single_layer_perceptron('MLContInput.csv', False)
Perceptron.multi_layer_perceptron('MLContInput.csv', False)
Perceptron.multiple_perceptron('MLContInput.csv', False)

Analysis.analysis('Perceptron-SingleLayer-Prediction-80.csv', 'CSI-MaxLoad-Result-80.csv', 'Perceptron-SingleLayer-Performance-80.txt')
Analysis.analysis('Perceptron-MultiLayer-Prediction-80.csv', 'CSI-MaxLoad-Result-80.csv', 'Perceptron-MultiLayer-Performance-80.txt')
Analysis.analysis('Perceptron-MultipleModel-Prediction-80.csv', 'CSI-MaxLoad-Result-80.csv', 'Perceptron-MultipleModel-Performance-80.txt')

Analysis.analysis('Perceptron-SingleLayer-5D-Prediction-80.csv', 'CSI-MaxLoad-Result-80.csv', 'Perceptron-SingleLayer-5D-Performance-80.txt')
Analysis.analysis('Perceptron-MultiLayer-5D-Prediction-80.csv', 'CSI-MaxLoad-Result-80.csv', 'Perceptron-MultiLayer-5D-Performance-80.txt')
Analysis.analysis('Perceptron-MultipleModel-5D-Prediction-80.csv', 'CSI-MaxLoad-Result-80.csv', 'Perceptron-MultipleModel-5D-Performance-80.txt')
