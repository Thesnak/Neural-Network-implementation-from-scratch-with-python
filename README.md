# Neural Network implementation from scratch with python


Neural Network with 2 Layers
----------------------------

This is a Python implementation of a simple 2-layer neural network. The network is designed to work with binary classification problems. The network consists of an input layer, a hidden layer with ReLU activation function and an output layer with sigmoid activation function.

The purpose of this code is to demonstrate the backpropagation algorithm for training a neural network with stochastic gradient descent.

### Installation

To use this code, you need to have the following libraries installed:

*   `numpy`
*   `matplotlib`

### Usage

Here is an example of how to use the `NeuralNet` class:

python

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from neuralnet import NeuralNet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# add header names
headers =  ['age', 'sex','chest_pain','resting_blood_pressure',  
        'serum_cholestoral', 'fasting_blood_sugar', 'resting_ecg_results',
        'max_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak',"slope of the peak",
        'num_of_major_vessels','thal', 'heart_disease']

heart_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat', sep=' ', names=headers)


import numpy as np
import warnings
warnings.filterwarnings("ignore") #suppress warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#convert imput to numpy arrays
X = heart_df.drop(columns=['heart_disease'])

#replace target class with 0 and 1 
#1 means "have heart disease" and 0 means "do not have heart disease"
heart_df['heart_disease'] = heart_df['heart_disease'].replace(1, 0)
heart_df['heart_disease'] = heart_df['heart_disease'].replace(2, 1)

y_label = heart_df['heart_disease'].values.reshape(X.shape[0], 1)

#split data into train and test set
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_label, test_size=0.2, random_state=2)

#standardize the dataset
sc = StandardScaler()
sc.fit(Xtrain)
Xtrain = sc.transform(Xtrain)
Xtest = sc.transform(Xtest)

print(f"Shape of train set is {Xtrain.shape}")
print(f"Shape of test set is {Xtest.shape}")
print(f"Shape of train label is {ytrain.shape}")
print(f"Shape of test labels is {ytest.shape}")


# Create a neural network object and fit it to the training data

nn = NeuralNet(layers=[13,10,1], learning_rate=0.01, iterations=500) # create the NN model
nn.fit(Xtrain, ytrain) #train the model


# Make predictions on the test data
train_pred = nn.predict(Xtrain)
test_pred = nn.predict(Xtest)

# Calculate the accuracy of the predictions
print("Train accuracy is {}".format(nn.acc(ytrain, train_pred)))
print("Test accuracy is {}".format(nn.acc(ytest, test_pred)))

# Plot the loss curve
nn.plot_loss()
```

### Class API

The `NeuralNet` class has the following methods:

*   `__init__(self, layers=[13,8,1], learning_rate=0.001, iterations=100)`: Constructor for the neural network object. `layers` is a list of integers specifying the number of neurons in each layer. The default value is `[13,8,1]`, which means that the input layer has 13 neurons, the hidden layer has 8 neurons, and the output layer has 1 neuron. `learning_rate` is a float specifying the learning rate of the stochastic gradient descent optimizer. The default value is `0.001`. `iterations` is an integer specifying the number of training iterations. The default value is `100`.
*   `init_weights(self)`: Initializes the weights and biases of the neural network using a random normal distribution.
*   `relu(self, Z)`: Computes the ReLU activation function.
*   `dRelu(self, x)`: Computes the derivative of the ReLU activation function.
*   `eta(self, x)`: Clips the value of `x` to avoid NaNs in the logarithm calculation.
*   `sigmoid(self, Z)`: Computes the sigmoid activation function.
*   `entropy_loss(self, y, yhat)`: Computes the cross-entropy loss function.
*   `forward_propagation(self)`: Performs the forward propagation step of the backpropagation algorithm.
*   `back_propagation(self, yhat)`: Performs the backpropagation step of the backpropagation algorithm and updates the weights and biases of the neural network.
*   `fit(self, X, y)`: Trains the neural network using the specified data and labels.
*   `predict(self, X)`: Predicts the labels of the input data using the trained neural network.
*   `acc(self, y, yhat)`: Calculates the accuracy of the predicted labels compared to the true labels.
*   `plot_loss(self)`: Plots the loss curve of the trained neural network.

### Contact Me


[My portfolio](https://bit.ly/3vCNonG)


### License

This code is released under the MIT License.

---
