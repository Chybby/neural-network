# Neural Network Project

As part of a uni course on machine learning, we had to choose a final project.
Since I found neural networks really interesting and wanted to learn more, I
chose to implement this neural network from scratch using Python and numpy.

I learned a lot from the Stanford CS class
[CS231n: Convolutional Neural Networks for Visual Recognition.](https://cs231n.github.io/)

## Requirements
- Python 3
- numpy
- matplotlib


## Creating the neural network
```python
nnet = NeuralNetwork(layer_sizes=[8, 100, 1],
                     activation=NeuralNetwork.RELU,
                     loss=NeuralNetwork.HINGE,
                     reg=NeuralNetwork.L2,
                     reg_strength=0.001
                     dropout_prob=0.5,
                     learning_task=NeuralNetwork.CLASSIFICATION,
                     learning_rate=0.00001,
                     learning_rate_decay=0.5,
                     learning_rate_epochs_per_decay=500
                     param_update_method=NeuralNetwork.NESTEROV_MOMENTUM,
                     momentum=0.9,
                     momentum_build=0.1,
                     momentum_epochs_per_build=500)
```
This creates a neural net with 8 input neurons, a single hidden layer of
100 neurons and a single output dimension.
Other possible argument options can be seen at the top of the NeuralNetwork
class


## Training the neural network
```python
nnet.train(train_X, train_y, 10000, batch_size=100,
           monitoring_flags=[NeuralNetwork.PRINT_TRAIN_LOSS,
                             NeuralNetwork.GRAPH_TRAIN_LOSS,
                             NeuralNetwork.PRINT_VALIDATION_LOSS,
                             NeuralNetwork.GRAPH_VALIDATION_LOSS],
           monitoring_resolution=100,
           val_X=val_X, val_y=val_y)
```
This trains the neural net using train_X and train_y for 10000 epochs with a
batch size of 100 between parameter updates.
This will also print and graph the training and validation loss 100 times
throughout the training.


## Using the neural network for predictions
```python
predictions = nnet.predict(test_X)
```

## Examples

### autoMpg

train_autompg.py contains an example of using the neural network to learn the
[autoMpg dataset](https://archive.ics.uci.edu/ml/datasets/auto+mpg)
from the UCI Machine Learning Repository.

### MNIST

train_mnist.py contains an example of using the neural network to learn the
[MNIST dataset](http://yann.lecun.com/exdb/mnist/) of handwritten
numbers.

### Binary functions

train_binary.py contains an example of using the neural network to learn a
couple 8-input binary functions, one linearly seperable and the other not.
This also shows that the neural network can also learn a linear model with no
hidden layers.
