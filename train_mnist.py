import numpy as np
import struct

from neural_network import NeuralNetwork

mnist_num_instances = None
mnist_num_inputs = None
mnist_num_outputs = 10

mnist_data = []
mnist_labels = []

# Read the data
with open('mnist-labels', 'rb') as f:
    magic = struct.unpack('>i', f.read(4))[0]
    num_labels = struct.unpack('>i', f.read(4))[0]
    for i in range(num_labels):
        label = struct.unpack('b', f.read(1))[0]
        mnist_labels.append(label)

with open('mnist-images', 'rb') as f:
    magic = struct.unpack('>i', f.read(4))[0]
    num_images = struct.unpack('>i', f.read(4))[0]
    mnist_num_instances = num_images
    num_rows = struct.unpack('>i', f.read(4))[0]
    num_columns = struct.unpack('>i', f.read(4))[0]
    mnist_num_inputs = num_rows * num_columns
    for image in range(num_images):
        pixels = struct.unpack('B' * num_columns * num_rows,
                               f.read(num_columns * num_rows))
        image_data = list(pixels)
        mnist_data.append(image_data)

X = np.zeros((mnist_num_instances, mnist_num_inputs))
y = np.zeros((mnist_num_instances, 1), dtype='int8')

for i, label in enumerate(mnist_labels):
    y[i] = label

for i, pixels in enumerate(mnist_data):
    X[i] = pixels

mnist_test_split = 0.1
mnist_num_test_instances = int(mnist_num_instances * mnist_test_split)
mnist_validation_split = 0.1
mnist_num_validation_instances = int(
    (mnist_num_instances - mnist_num_test_instances) * mnist_validation_split
)

# Split into train, validation and test instances
indices = np.random.permutation(mnist_num_instances)
test_indices = indices[:mnist_num_test_instances]
rest_indices = indices[mnist_num_test_instances:]
validation_indices = rest_indices[:mnist_num_validation_instances]
train_indices = rest_indices[mnist_num_validation_instances:]
train_X, val_X, test_X = (X[train_indices,:], X[validation_indices,:],
                          X[test_indices,:])
train_y, val_y, test_y = (y[train_indices], y[validation_indices],
                          y[test_indices])

# Train a linear classifier
nnet = NeuralNetwork(layer_sizes=[mnist_num_inputs, 10],
                     dropout_prob=0.5,
                     loss=NeuralNetwork.HINGE,
                     learning_task=NeuralNetwork.CLASSIFICATION,
                     reg_strength=0.001,
                     learning_rate=0.00000001,
                     learning_rate_decay=0.0,
                     learning_rate_epochs_per_decay=500)

nnet.train(train_X, train_y, 100, batch_size=100,
           monitoring_flags=NeuralNetwork.MONITOR_ALL,
           val_X=val_X, val_y=val_y)

# Get predictions for test set
correct_predictions = 0
predictions = nnet.predict(test_X)
for i in range(predictions.shape[0]):
    if predictions[i] == test_y[i]:
        correct_predictions += 1
    else:
        print('Incorrect prediction: y=%d (should be %d)' % (
              predictions[i], test_y[i]))

print('Correctly predicting %d%% of instances' % (correct_predictions * 100 /
                                                  mnist_num_test_instances))


# Train a neural net
nnet = NeuralNetwork(layer_sizes=[mnist_num_inputs, 300, 10],
                     dropout_prob=0.5,
                     loss=NeuralNetwork.HINGE,
                     learning_task=NeuralNetwork.CLASSIFICATION,
                     reg_strength=0.00,
                     learning_rate=0.00000005,
                     learning_rate_decay=0.0,
                     learning_rate_epochs_per_decay=500)

nnet.train(train_X, train_y, 100, batch_size=1000,
           monitoring_flags=NeuralNetwork.MONITOR_ALL,
           monitoring_resolution=20,
           val_X=val_X, val_y=val_y)

# Get predictions for test set
correct_predictions = 0
predictions = nnet.predict(test_X)
for i in range(predictions.shape[0]):
    if predictions[i] == test_y[i]:
        correct_predictions += 1
    else:
        print('Incorrect prediction: y=%d (should be %d)' % (
              predictions[i], test_y[i]))

print('Correctly predicting %d%% of instances' % (correct_predictions * 100 /
                                                  mnist_num_test_instances))
