import numpy as np

from neural_network import NeuralNetwork

autompg_data = []
discrete_attribute_values = {
    0 : [3, 4, 5, 6, 8],
    5 : [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82],
    6 : [1, 2, 3]
}

# Training on autoMpg
with open('autoMpgData.csv') as f:
    for line in f:
        if '?' in line:
            # Skip instances with missing attributes
            continue
        attributes = list(map(float, line.strip().split(',')))
        autompg_data.append(attributes)

autompg_num_inputs = (len(autompg_data[0]) - 1 -
                      len(discrete_attribute_values) +
                      sum(map(len, discrete_attribute_values.values())))
autompg_num_instances = len(autompg_data)
autompg_test_split = 0.1
autompg_num_test_instances = int(autompg_num_instances * autompg_test_split)
autompg_validation_split = 0.1
autompg_num_validation_instances = int(
    (autompg_num_instances - autompg_num_test_instances) *
    autompg_validation_split
)

# Gather the data
X = np.zeros((autompg_num_instances, autompg_num_inputs))
y = np.zeros(autompg_num_instances)
for i, instance in enumerate(autompg_data):
    converted_attributes = []

    # Convert class attributes into indicator attributes
    for j, attribute in enumerate(instance):
        if j in discrete_attribute_values:
            attribute = int(attribute)
            converted_attributes += [
                int(x == attribute) for x in discrete_attribute_values[j]
            ]
        else:
            converted_attributes.append(attribute)

    X[i] = converted_attributes[:-1]
    y[i] = converted_attributes[-1]

# Zero-center data
X -= np.mean(X, axis=0)
# Normalize data
X /= np.std(X, axis=0)

# Split into train, validation and test instances
indices = np.random.permutation(autompg_num_instances)
test_indices = indices[:autompg_num_test_instances]
rest_indices = indices[autompg_num_test_instances:]
validation_indices = rest_indices[:autompg_num_validation_instances]
train_indices = rest_indices[autompg_num_validation_instances:]
train_X, val_X, test_X = (X[train_indices,:], X[validation_indices,:],
                          X[test_indices,:])
train_y, val_y, test_y = (y[train_indices], y[validation_indices],
                          y[test_indices])

# Train a linear regression
nnet = NeuralNetwork(layer_sizes=[autompg_num_inputs, 1],
                     dropout_prob=1,
                     loss=NeuralNetwork.SQUARED_ERROR,
                     learning_task=NeuralNetwork.REGRESSION,
                     reg_strength=0.01,
                     learning_rate=0.00001,
                     learning_rate_decay=0.0,
                     learning_rate_epochs_per_decay=500)

nnet.train(train_X, train_y, 100,
           monitoring_flags=[NeuralNetwork.PRINT_TRAIN_LOSS,
                             NeuralNetwork.GRAPH_TRAIN_LOSS,
                             NeuralNetwork.PRINT_VALIDATION_LOSS,
                             NeuralNetwork.GRAPH_VALIDATION_LOSS],
           val_X=val_X, val_y=val_y)

# Calculate the error on the test data
squared_errors = 0
predictions = nnet.predict(test_X)
for i in range(predictions.shape[0]):
    squared_errors += (predictions[i] - test_y[i])**2

print('Average test squared error: %f' % (squared_errors /
                                          autompg_num_test_instances))


# Train a neural net
nnet = NeuralNetwork(layer_sizes=[autompg_num_inputs, 100, 1],
                     dropout_prob=0.5,
                     loss=NeuralNetwork.SQUARED_ERROR,
                     learning_task=NeuralNetwork.REGRESSION,
                     reg_strength=0.01,
                     learning_rate=0.000001,
                     learning_rate_decay=0.0,
                     learning_rate_epochs_per_decay=500)

nnet.train(train_X, train_y, 100,
           monitoring_flags=[NeuralNetwork.PRINT_TRAIN_LOSS,
                             NeuralNetwork.GRAPH_TRAIN_LOSS,
                             NeuralNetwork.PRINT_VALIDATION_LOSS,
                             NeuralNetwork.GRAPH_VALIDATION_LOSS],
           val_X=val_X, val_y=val_y)

# Calculate the error on the test data
squared_errors = 0
predictions = nnet.predict(test_X)
for i in range(predictions.shape[0]):
    squared_errors += (predictions[i] - test_y[i])**2

print('Average test squared error: %f' % (squared_errors /
                                          autompg_num_test_instances))
