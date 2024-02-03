# NEURAL NETWORK UTILITIES

Module containg functions and tools for constructing a Neural Network and perform forward propagation.

**Utils.py contains the following:**
- DenseLayer: Construction of a Dense layer of the Network
- ActivationReLU: Rectified Linear activation class as follows - _max(0,X)_
- ActivationSoftMax: Soft Max activation class - _exp(X)/sum(exp(X))_ where X is an array of inputs.
- Loss: Parent class for loss calculations.
- CCE: function to calculate Catagorical Cross-Entropy.

Utils.py also contains an  example script showing how above tools can be chained to perfom contruction and forward propagation.

Note: Utils.py requires Numpy Python library
