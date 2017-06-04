import math
import matplotlib.pyplot as plt
import numpy as np

class NeuralNetwork():

    # Activation functions
    RELU = 0
    TANH = 1
    SIGMOID = 2

    # Loss functions
    CROSS_ENTROPY = 0
    HINGE = 1
    SQUARED_ERROR = 2

    # Regularization functions
    L2 = 0
    L1 = 1

    # Learning tasks
    CLASSIFICATION = 0
    ATTRIBUTE_CLASSIFICATION = 1
    REGRESSION = 2

    # Parameter update methods
    VANILLA_SGD = 0
    MOMENTUM = 1
    NESTEROV_MOMENTUM = 2

    # Monitoring flags
    PRINT_LOSS = 0
    PRINT_VAL_LOSS = 1
    GRAPH_LOSS = 2
    GRAPH_VAL_LOSS = 3

    def __init__(self, layer_sizes,
                 activation=RELU,
                 loss=CROSS_ENTROPY,
                 reg=L2, reg_strength=0.001,
                 dropout_prob=0.5,
                 learning_task=CLASSIFICATION,
                 learning_rate=1, learning_rate_decay=0,
                 learning_rate_epochs_per_decay=500,
                 param_update_method=NESTEROV_MOMENTUM,
                 momentum=0.9, momentum_build=0,
                 momentum_epochs_per_build=500):

        # Matrices containing weights between neural network layers
        self.W = []
        self.b = []
        # Matrices containing derivatives of weights
        self.dW = []
        self.db = []

        # Denominator for the recommended initialization distribution
        init_denom = math.sqrt
        if activation == NeuralNetwork.RELU:
            # relu specifically has a different recommended denominator
            init_denom = lambda n: 1/math.sqrt(2/n)

        for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.W.append(
                # Recommended initialization distribution
                np.random.randn(input_size, output_size) / init_denom(input_size)
            )
            self.dW.append(np.zeros((input_size, output_size)))

            self.b.append(np.zeros((1, output_size)))
            self.db.append(np.zeros((1, output_size)))

        # Learning rate
        self.learning_rate = learning_rate
        # How much to decay the learning rate by
        self.learning_rate_decay = learning_rate_decay
        # How many epochs to train between decaying the learning rate
        self.learning_rate_epochs_per_decay = learning_rate_epochs_per_decay

        # Regularization strength
        self.reg_param = reg_strength

        # Dropout probability, the probability that a neuron will be kept alive
        # during one training pass
        self.dropout_prob = dropout_prob

        # Parameter update method
        self.param_update_method = param_update_method
        if (param_update_method == NeuralNetwork.NESTEROV_MOMENTUM or
            param_update_method == NeuralNetwork.MOMENTUM):
            self.Wv = []
            self.bv = []
            for input_size, output_size in zip(layer_sizes[:-1],
                                               layer_sizes[1:]):
                self.Wv.append(np.zeros((input_size, output_size)))
                self.bv.append(np.zeros((1, output_size)))
        elif param_update_method != NeuralNetwork.VANILLA_SGD:
            raise ValueError('Invalid parameter update method')


        # Momentum update mu parameter
        self.momentum = momentum
        # How much to build the momentum by
        self.momentum_build = momentum_build
        # How many epochs to train between building the momentum
        self.momentum_epochs_per_build = momentum_epochs_per_build

        # Node activation function and its derivative
        if activation == NeuralNetwork.RELU:
            self.activation = self._relu
            self.d_activation = self._d_relu
        elif activation == NeuralNetwork.SIGMOID:
            self.activation = self._sigmoid
            self.d_activation = self._d_sigmoid
        elif activation == NeuralNetwork.TANH:
            self.activation = self._tanh
            self.d_activation = self._d_tanh
        else:
            raise ValueError('Invalid activation function')

        # What type of learning this net should perform
        # Standard classification is when each instance has exactly one correct
        # class
        # Attribute classification is when each instance can be of more than one
        # class
        # Regression is when each instance is associated with 1 or more real
        # values, not discrete classes
        self.learning_task = learning_task
        if learning_task == NeuralNetwork.REGRESSION:
            # Default to squared error loss when performing regression
            loss = NeuralNetwork.SQUARED_ERROR
            # Don't perform dropout
            self.dropout_prob = 1
        elif (learning_task != NeuralNetwork.CLASSIFICATION and
              learning_task != NeuralNetwork.ATTRIBUTE_CLASSIFICATION):
            raise ValueError('Invalid learning task')

        # Loss function
        self.loss = loss
        if learning_task == NeuralNetwork.CLASSIFICATION:
            if loss == NeuralNetwork.CROSS_ENTROPY:
                self.data_loss = self._cross_entropy_loss
            elif loss == NeuralNetwork.HINGE:
                self.data_loss = self._hinge_loss
            else:
                raise ValueError('Invalid loss function')
        elif learning_task == NeuralNetwork.ATTRIBUTE_CLASSIFICATION:
            if loss == NeuralNetwork.CROSS_ENTROPY:
                self.data_loss = self._cross_entropy_loss_attr
            elif loss == NeuralNetwork.HINGE:
                self.data_loss = self._hinge_loss_attr
            else:
                raise ValueError('Invalid loss function')
        elif learning_task == NeuralNetwork.REGRESSION:
            if loss == NeuralNetwork.SQUARED_ERROR:
                self.data_loss = self._squared_error_loss
            else:
                raise ValueError('Invalid loss function')

        # Regularization function and its derivative
        if reg == NeuralNetwork.L2:
            self.reg_loss = self._l2_reg
            self.d_reg_loss = self._d_l2_reg
        elif reg == NeuralNetwork.L1:
            self.reg_loss = self._l1_reg
            self.d_reg_loss = self._d_l1_reg
        else:
            raise ValueError('Invalid regularization function')

        # Since a lot of values calculated in forward propagation are needed in
        # back propagation, cache them to avoid recalculating them
        # Each layer has a dict that can be used to store values
        self._cache = [{} for i in range(len(self.W))]

        # Which layer training is at, either in forward or back propagation
        self._current_layer = 0


    # Returns the cache for the current layer
    @property
    def _curr_cache(self):
        return self._cache[self._current_layer]


    # Trains the neural network with some training instances for a number of
    # epochs
    #
    # X is a (N x I) matrix where N is the number of training instances and I is
    # the number of attributes
    # Each value is an attribute value
    #
    # If the learning task is classification:
    #   Y is a (N x 1) matrix where N is the number of training instances
    #   Each value is the index of the correct class for that training instance
    #   eg. y[3][0] = 3 means the 4th training instance is of the 4th class
    #
    # If the learning task is attribute classification:
    #   Y is a (N x C) matrix where N is the number of training instances and C
    #   is the number of classes
    #   Each value is either 0 or 1, indicating whether that training instance
    #   falls into that class
    #   eg. y[3] = [0, 1, 1] means the 4th training instance is of the 2nd and
    #   3rd classes but not of the 1st class
    #
    # If the learning task is regression:
    #   Y is a (N x 1) matrix where N is the number of training instances
    #   Each value is the real value associated with that training  instance
    #   eg. y[3] = [0.49] means the 4th training instance is associated with the
    #   real value 0.49
    #
    # Please ensure that X and y are signed
    def train(self, X, y, epochs=1000, batch_size=None,
              val_X=None, val_y=None, monitoring_flags=[],
              monitoring_resolution=100):
        num_instances = X.shape[0]

        graph_loss = NeuralNetwork.GRAPH_LOSS in monitoring_flags
        graph_val_loss = NeuralNetwork.GRAPH_VAL_LOSS in monitoring_flags
        print_loss = NeuralNetwork.PRINT_LOSS in monitoring_flags
        print_val_loss = NeuralNetwork.PRINT_VAL_LOSS in monitoring_flags

        loss_plotline = None
        val_loss_plotline = None

        if ((graph_val_loss or print_val_loss) and
           (val_X is None or val_y is None)):
            raise ValueError('Attempting to monitor validation loss with no '
                             'validation instances')

        if graph_loss:
            loss_plotline,  = plt.plot([], [], 'r-', label='Training loss')
        if graph_val_loss:
            val_loss_plotline, = plt.plot([], [], 'b-', label='Validation loss')

        if graph_loss or graph_val_loss:
            plt.ion()
            plt.xlim([0, epochs])
            plt.title("Loss per Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid()
            plt.show()

        if (self.learning_task == NeuralNetwork.ATTRIBUTE_CLASSIFICATION and
                     self.loss == NeuralNetwork.HINGE):
            # Turn y from 0s and 1s into -1s and 1s
            y = y.copy()*2 - 1
            val_y = val_y.copy()*2 - 1

        # Default to including all instances in a batch
        if batch_size == None:
            batch_size = num_instances

        for epoch in range(epochs):
            if epoch != 0:
                if epoch % self.learning_rate_epochs_per_decay == 0:
                    # Decay the learning rate
                    decay_by = self.learning_rate * self.learning_rate_decay
                    self.learning_rate -= decay_by

                if epoch % self.momentum_epochs_per_build == 0:
                    # Build the momentum
                    build_by = (1 - self.momentum) * self.momentum_build
                    self.momentum += build_by

            for batch in range(num_instances//batch_size):
                # Sample a batch of size batch_size
                if batch_size == num_instances:
                    X_batch = X
                    y_batch = y
                else:
                    sample = np.random.choice(num_instances, batch_size,
                                              replace=False)
                    X_batch = X[sample, :]
                    y_batch = y[sample]

                # Feed through network to calculate scores
                scores, last_layer_output = self._calculate_scores(
                        X_batch,
                        dropout_prob=self.dropout_prob,
                        cache_results=True
                )
                loss, dscores = self._loss(scores, y_batch)

                val_loss = None
                if print_val_loss or graph_val_loss:
                    # Calculate the validation set loss
                    val_scores, _ = self._calculate_scores(val_X)
                    val_loss, _ = self._loss(val_scores, val_y)

                if epoch % (epochs//monitoring_resolution) == 0:
                    # Print and graph loss and validation loss
                    if print_loss:
                        print('Training loss for epoch %d, batch %d: %g' % (
                                epoch, batch, loss))
                    if print_val_loss:
                        print('Validation loss for epoch %d, batch %d: %g' % (
                                epoch, batch, val_loss))

                    if graph_loss:
                        loss_plotline.set_xdata(
                            np.append(loss_plotline.get_xdata(), epoch)
                        )
                        loss_plotline.set_ydata(
                            np.append(loss_plotline.get_ydata(), loss)
                        )

                    if graph_val_loss:
                        val_loss_plotline.set_xdata(
                            np.append(val_loss_plotline.get_xdata(), epoch)
                        )
                        val_loss_plotline.set_ydata(
                            np.append(val_loss_plotline.get_ydata(), val_loss)
                        )

                    if graph_loss or graph_val_loss:
                        ax = plt.gca()
                        ax.relim()
                        ax.autoscale_view()
                        plt.draw()
                        plt.pause(0.00000001)

                # Back propagate from the output layer
                dWoutput = np.dot(last_layer_output.T, dscores)
                dWoutput += self.d_reg_loss(self.W[-1])
                dboutput = np.sum(dscores, axis=0, keepdims=True)

                self.dW[self._current_layer] = dWoutput
                self.db[self._current_layer] = dboutput

                self._curr_cache['ddot'] = dscores

                # Back propagate back through the hidden layers
                for W, b in zip(self.W[:0:-1], self.b[:0:-1]):
                    prev_ddot = self._curr_cache['ddot']
                    self._current_layer -= 1

                    dot = self._curr_cache['dot']
                    layer_input = self._curr_cache['layer_input']

                    dlayer_output = np.dot(prev_ddot, W.T)
                    ddot = dlayer_output * self.d_activation(dot)

                    dW = np.dot(layer_input.T, ddot)
                    dW += self.d_reg_loss(self.W[self._current_layer])
                    db = np.sum(ddot, axis=0, keepdims=True)

                    self._curr_cache['ddot'] = ddot
                    self.dW[self._current_layer] = dW
                    self.db[self._current_layer] = db

                # Update weights
                self._update_weights()

        if graph_loss or graph_val_loss:
            plt.ioff()
            plt.show()


    # Feeds forward through the network to calculate the scores for some input
    def _calculate_scores(self, X, dropout_prob=1, cache_results=False):
        self._current_layer = 0
        # Feed through hidden layers
        layer_output = X
        for W, b in zip(self.W[:-1], self.b[:-1]):
            layer_input = layer_output

            dot = np.dot(layer_input, W) + b
            layer_output = self.activation(dot, cache_results)

            if dropout_prob != 1:
                dropout_mask = np.random.rand(*layer_output.shape)
                dropout_mask = dropout_mask < dropout_prob
                dropout_mask = dropout_mask / dropout_prob

                layer_output *= dropout_mask

            if cache_results:
                self._curr_cache['dot'] = dot
                self._curr_cache['layer_input'] = layer_input

            self._current_layer += 1

        # Feed forward through the output layer
        scores = np.dot(layer_output, self.W[-1]) + self.b[-1]
        return scores, layer_output


    # Updates weights with gradients calculated in training
    def _update_weights(self):
        for layer in range(len(self.W)):
            if self.param_update_method == NeuralNetwork.VANILLA_SGD:
                self.W[layer] -= self.learning_rate * self.dW[layer]
                self.b[layer] -= self.learning_rate * self.db[layer]
            elif self.param_update_method == NeuralNetwork.MOMENTUM:
                W = self.W[layer]
                dW = self.dW[layer]
                Wv = self.Wv[layer]

                Wv = self.momentum * Wv - self.learning_rate * dW
                W += Wv

                self.Wv[layer] = Wv

                b = self.b[layer]
                db = self.db[layer]
                bv = self.bv[layer]

                bv = self.momentum * bv - self.learning_rate * db
                b += bv

                self.bv[layer] = bv
            elif self.param_update_method == NeuralNetwork.NESTEROV_MOMENTUM:
                W = self.W[layer]
                dW = self.dW[layer]
                Wv = self.Wv[layer]
                Wv_prev = Wv.copy()

                Wv = self.momentum * Wv - self.learning_rate * dW
                W += -self.momentum * Wv_prev + (1 + self.momentum) * Wv

                self.Wv[layer] = Wv

                b = self.b[layer]
                db = self.db[layer]
                bv = self.bv[layer]
                bv_prev = bv.copy()

                bv = self.momentum * bv - self.learning_rate * db
                b += -self.momentum * bv_prev + (1 + self.momentum) * bv

                self.bv[layer] = bv


    # Predicts classes for the given test instances
    #
    # If the learning task is classification:
    #   Returns the index of the predicted class for each test instance
    #   eg. prediction[3] = 2 indicates that the 4th test instance is predicted
    #   to be of the 3rd class
    # If the learning task is attribute classification:
    #   Returns whether a test instance is predicted to be of a certain class
    #   for each test instance and for each class
    #   eg. prediction[3] = [1, 0, 1] indicates that the 4th test instance is
    #   predicted to be of the 1st and 3rd classes but not of the 2nd class
    # If the learning task is regression:
    #   Returns the predicted real value for each test instance
    #   eg. prediction[3] = 0.49 indicates that the real value associated with
    #   the 4th test instance is predicted to be 0.49
    def predict(self, X):
        scores, _ = self._calculate_scores(X)

        prediction = None

        if self.learning_task == NeuralNetwork.CLASSIFICATION:
            prediction = np.argmax(scores, axis=1)
        elif self.learning_task == NeuralNetwork.ATTRIBUTE_CLASSIFICATION:
            # Assume non-classification on 0 score
            scores[scores == 0] = -1
            # Turn a matrix of negatives and positives into a matrix of
            # 0s and 1s
            prediction = (np.sign(scores) + 1)/2
        elif self.learning_task == NeuralNetwork.REGRESSION:
            prediction = scores
        return prediction


    # Averages all data losses and combines them with the regularization losses
    # to calculate the final loss for an epoch
    def _loss(self, S, y, cache_results=False):
        losses, dscores = self.data_loss(S, y)
        loss = np.sum(losses)/np.size(losses)
        for W in self.W:
            loss += self.reg_loss(W)
        return loss, dscores


    # The cross-entropy loss function, also calculates dloss/dscores
    def _cross_entropy_loss(self, S, y):
        probs = np.exp(S)/np.sum(np.exp(S), axis=1, keepdims=True)

        neg_log_probs = -np.log(probs)
        result = neg_log_probs[range(S.shape[0]), y]

        num_instances = probs.shape[0]
        probs[range(num_instances), y] -= 1

        return result, probs/num_instances


    # The cross-entropy loss function when doing attribute classification
    # Also calculates dloss/dscores
    def _cross_entropy_loss_attr(self, S, y):
        sig = self._sigmoid(S)
        probs = y*np.log(sig) + (1 - y)*np.log(1 - sig)
        return np.sum(-probs, axis=1), -y + sig


    # The hinge loss function, also calculates dloss/dscores
    def _hinge_loss(self, S, y):
        num_instances = S.shape[0]
        margins = np.maximum(0,
            S - S[range(num_instances), y].reshape(num_instances, 1) + 1)
        margins[range(margins.shape[0]),y] = 0

        dS = np.sign(margins)
        dS[range(dS.shape[0]), y] = -np.sum(dS, axis=1)

        return np.sum(margins, axis=1), dS


    # The hinge loss function when doing attribute classification
    # Also calculates dloss/dscores
    def _hinge_loss_attr(self, S, y):
        margins = np.maximum(0, 1 - S*y)
        return np.sum(margins, axis=1), -y * np.sign(margins)


    # The squared error loss function, also calculates dloss/dscores
    def _squared_error_loss(self, S, y):
        diffs = S - y.reshape(y.shape[0], 1)
        return np.sum(diffs**2, axis=1)**2, 2*diffs


    # The L2 regularization function
    def _l2_reg(self, W):
        return 0.5 * self.reg_param * np.sum(W * W)


    # The derivative of the L2 regularization function
    def _d_l2_reg(self, W):
        return self.reg_param * W


    # The L1 regularization function
    def _l1_reg(self, W):
        return self.reg_param * np.sum(np.abs(W))


    # The derivative of the L1 regularization function
    def _d_l1_reg(self, W):
        return self.reg_param * np.sign(W)


    # The sigmoid activation function
    def _sigmoid(self, X, cache_results=False):
        result = 1/(1 + np.exp(-X))
        if cache_results:
            # Save this result as it is used during back propagation
            self._curr_cache['sigmoid_result'] = result
        return result


    # The derivative of the sigmoid activation function
    def _d_sigmoid(self, X):
        # Assumes that the sigmoid function was previously called on this layer
        # in the forward pass of this epoch
        prev_result = self._curr_cache['sigmoid_result']

        return prev_result*(1 - prev_result)


    # The tanh activation function
    def _tanh(self, X, cache_results=False):
        result = np.tanh(X)
        if cache_results:
            # Save this result as it is used during back propagation
            self._curr_cache['tanh_result'] = result
        return result


    # The derivative of the tanh activation function
    def _d_tanh(self, X):
        # Assumes that the tanh function was previously called on this layer
        # in the forward pass of this epoch
        prev_result = self._curr_cache['tanh_result']

        return 1 - prev_result**2


    # The relu activation function
    def _relu(self, X, cache_results=False):
        return np.maximum(0, X)


    # The derivative of the relu activation function
    def _d_relu(self, X):
        result = X.copy()
        result[result <= 0] = 0
        result[result > 0] = 1
        return result



#####
# This section tests the implementation on a few example data sets
#####

import itertools

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

autompg_num_inputs = len(autompg_data[0]) - 1 - len(discrete_attribute_values) + sum(map(len, discrete_attribute_values.values()))
autompg_num_instances = len(autompg_data)
autompg_test_split = 0.1
autompg_num_test_instances = int(autompg_num_instances*autompg_test_split)
autompg_validation_split = 0.1
autompg_num_validation_instances = int((autompg_num_instances-autompg_num_test_instances)*autompg_validation_split)

# Gather the data
X = np.zeros((autompg_num_instances, autompg_num_inputs))
y = np.zeros(autompg_num_instances)
for i, instance in enumerate(autompg_data):
    converted_attributes = []

    # Convert class attributes into indicator attributes
    for j, attribute in enumerate(instance):
        if j in discrete_attribute_values:
            attribute = int(attribute)
            converted_attributes += [int(x == attribute) for x in discrete_attribute_values[j]]
        else:
            converted_attributes.append(attribute)

    print(converted_attributes)
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
train_X, val_X, test_X = X[train_indices,:], X[validation_indices,:], X[test_indices,:]
train_y, val_y, test_y = y[train_indices], y[validation_indices], y[test_indices]

# Train a linear regression
nnet = NeuralNetwork(layer_sizes=[autompg_num_inputs, 1],
                     dropout_prob=1,
                     loss=NeuralNetwork.SQUARED_ERROR,
                     learning_task=NeuralNetwork.REGRESSION,
                     reg_strength=0.01,
                     learning_rate=0.0000001,
                     learning_rate_decay=0.0,
                     learning_rate_epochs_per_decay=500)

nnet.train(train_X, train_y, 10000,
           monitoring_flags=[NeuralNetwork.PRINT_LOSS,
                             NeuralNetwork.GRAPH_LOSS,
                             NeuralNetwork.PRINT_VAL_LOSS,
                             NeuralNetwork.GRAPH_VAL_LOSS],
           val_X=val_X, val_y=val_y)

# Calculate the error on the test data
squared_error = 0
predictions = nnet.predict(test_X)
for i in range(predictions.shape[0]):
    #print('Prediction for x=%s: y=%f (should be %f)' % (test_X[i], predictions[i], test_y[i]))
    squared_error += (predictions[i] - test_y[i])**2

print('Average squared error: %f' % (squared_error/autompg_num_test_instances))

# Train a neural net
nnet = NeuralNetwork(layer_sizes=[autompg_num_inputs, 100, 1],
                     dropout_prob=1,
                     loss=NeuralNetwork.SQUARED_ERROR,
                     learning_task=NeuralNetwork.REGRESSION,
                     reg_strength=0.001,
                     learning_rate=0.00000001,
                     learning_rate_decay=0.0,
                     learning_rate_epochs_per_decay=500)

nnet.train(train_X, train_y, 10000,
           monitoring_flags=[NeuralNetwork.PRINT_LOSS,
                             NeuralNetwork.GRAPH_LOSS,
                             NeuralNetwork.PRINT_VAL_LOSS,
                             NeuralNetwork.GRAPH_VAL_LOSS],
           val_X=val_X, val_y=val_y)

# Calculate the error on the test data
squared_error = 0
predictions = nnet.predict(test_X)
for i in range(predictions.shape[0]):
    #print('Prediction for x=%s: y=%f (should be %f)' % (test_X[i], predictions[i], test_y[i]))
    squared_error += (predictions[i] - test_y[i])**2

print('Average squared error: %f' % (squared_error/autompg_num_test_instances))


# Training on binary function
binary_num_inputs = 8
binary_num_instances = 2**binary_num_inputs
binary_num_classes = 2

X = np.zeros((binary_num_instances, binary_num_inputs), dtype='int8')
y = np.zeros(binary_num_instances, dtype='int8')

for i, product in enumerate(itertools.product('01', repeat=binary_num_inputs)):
    for j, bit in enumerate(product):
        X[i, j] = int(bit)
    #y[i] = X[i, 0] & X[i, 1] & X[i, 2] & X[i, 3] & X[i, 4] & X[i, 5] & X[i, 6] & X[i, 7]
    y[i] = X[i, 0] ^ X[i, 1] ^ X[i, 2] ^ X[i, 3] ^ X[i, 4] ^ X[i, 5] ^ X[i, 6] ^ X[i, 7]

# Train a linear classifier
nnet = NeuralNetwork(layer_sizes=[binary_num_inputs, binary_num_classes],
                     dropout_prob=1,
                     loss=NeuralNetwork.HINGE,
                     learning_task=NeuralNetwork.CLASSIFICATION,
                     reg_strength=0,
                     learning_rate=0.000001,
                     learning_rate_decay=0.0,
                     learning_rate_epochs_per_decay=500)

nnet.train(X, y, 10000,
           monitoring_flags=[NeuralNetwork.PRINT_LOSS,
                             NeuralNetwork.GRAPH_LOSS,])

# Get predictions for every possible input
correct_predictions = 0
predictions = nnet.predict(X)
for i in range(predictions.shape[0]):
    if predictions[i] == y[i]:
        correct_predictions += 1
    else:
        print('Incorrect prediction: x=%s, y=%d (should be %d)' % (X[i], predictions[i], y[i]))

print('Correctly predicting %d%% of instances' % (correct_predictions*100/binary_num_instances))

# Train a neural net classifier
nnet = NeuralNetwork(layer_sizes=[binary_num_inputs, 100, binary_num_classes],
                     dropout_prob=1,
                     loss=NeuralNetwork.HINGE,
                     learning_task=NeuralNetwork.CLASSIFICATION,
                     reg_strength=0,
                     learning_rate=0.0001,
                     learning_rate_decay=0.0,
                     learning_rate_epochs_per_decay=500)

nnet.train(X, y, 2000,
           monitoring_flags=[NeuralNetwork.PRINT_LOSS,
                             NeuralNetwork.GRAPH_LOSS,])

# Get predictions for every possible input
correct_predictions = 0
predictions = nnet.predict(X)
for i in range(predictions.shape[0]):
    if predictions[i] == y[i]:
        correct_predictions += 1
    else:
        print('Incorrect prediction: x=%s, y=%d (should be %d)' % (X[i], predictions[i], y[i]))

print('Correctly predicting %d%% of instances' % (correct_predictions*100/binary_num_instances))
