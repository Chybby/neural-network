import itertools
import numpy as np

from neural_network import NeuralNetwork

# Training on binary function
binary_num_inputs = 8
binary_num_instances = 2**binary_num_inputs
binary_num_classes = 2

X = np.zeros((binary_num_instances, binary_num_inputs), dtype='int8')
y = np.zeros((binary_num_instances, 1), dtype='int8')

for i, product in enumerate(itertools.product('01', repeat=binary_num_inputs)):
    for j, bit in enumerate(product):
        X[i, j] = int(bit)

    # Linearly seperable
    y[i] = (X[i, 0] & X[i, 1] & X[i, 2] & X[i, 3] &
            X[i, 4] & X[i, 5] & X[i, 6] & X[i, 7])

    # Not linearly seperable
    #y[i] = (X[i, 0] ^ X[i, 1] ^ X[i, 2] ^ X[i, 3] ^
    #        X[i, 4] ^ X[i, 5] ^ X[i, 6] ^ X[i, 7])


# Train a linear classifier
nnet = NeuralNetwork(layer_sizes=[binary_num_inputs, binary_num_classes],
                     dropout_prob=1,
                     loss=NeuralNetwork.CROSS_ENTROPY,
                     learning_task=NeuralNetwork.CLASSIFICATION,
                     reg_strength=0,
                     learning_rate=3,
                     learning_rate_decay=0,
                     learning_rate_epochs_per_decay=5000)

nnet.train(X, y, 400,
           monitoring_flags=[NeuralNetwork.PRINT_TRAIN_LOSS,
                             NeuralNetwork.GRAPH_TRAIN_LOSS,
                             NeuralNetwork.PRINT_TRAIN_ACCURACY,
                             NeuralNetwork.GRAPH_TRAIN_ACCURACY])


# Get predictions for every possible input
correct_predictions = 0
predictions = nnet.predict(X)
for i in range(predictions.shape[0]):
    if predictions[i] == y[i]:
        correct_predictions += 1
    else:
        print('Incorrect prediction: x=%s, y=%d (should be %d)' % (
              X[i], predictions[i], y[i]))

print('Correctly predicting %d%% of instances' % (correct_predictions * 100 /
                                                  binary_num_instances))


# Train a neural net classifier
nnet = NeuralNetwork(layer_sizes=[binary_num_inputs, 100, binary_num_classes],
                     dropout_prob=1,
                     loss=NeuralNetwork.HINGE,
                     learning_task=NeuralNetwork.CLASSIFICATION,
                     reg_strength=0,
                     learning_rate=0.001,
                     learning_rate_decay=0.0,
                     learning_rate_epochs_per_decay=500)

nnet.train(X, y, 300,
           monitoring_flags=[NeuralNetwork.PRINT_TRAIN_LOSS,
                             NeuralNetwork.GRAPH_TRAIN_LOSS,
                             NeuralNetwork.PRINT_TRAIN_ACCURACY,
                             NeuralNetwork.GRAPH_TRAIN_ACCURACY])

# Get predictions for every possible input
correct_predictions = 0
predictions = nnet.predict(X)
for i in range(predictions.shape[0]):
    if predictions[i] == y[i]:
        correct_predictions += 1
    else:
        print('Incorrect prediction: x=%s, y=%d (should be %d)' % (
              X[i], predictions[i], y[i]))

print('Correctly predicting %d%% of instances' % (correct_predictions * 100 /
                                                  binary_num_instances))
