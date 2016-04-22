# startupMLChallenge

First step is to pre-process the data. This has been described as below

1. Aggregate all data together and shuffle them. Notebook [here](https://github.com/leo237/activity_recognition/blob/master/Notebooks/Pre-Processing%20.ipynb)
2. Divide the data for performing 4 cross-validation and testing data. Notebook [here](https://github.com/leo237/activity_recognition/blob/master/Notebooks/Pre%20Process%202.ipynb)

Tried to solve it using two ways.


1. Using Neural Network. Got accuracy of about 64%. 
 
    Notebook [here](https://github.com/leo237/activity_recognition/blob/master/Notebooks/Training%20New%20Data.ipynb) to train data for the first time. 

    To train using an existing model, we can use [this](https://github.com/leo237/activity_recognition/blob/master/Notebooks/Train%20with%20an%20existing%20model..ipynb) notebook.

2. Using Random Forest Classification method. Got accuracy of 71%.

    Notebook  [here](https://github.com/leo237/activity_recognition/blob/master/Notebooks/Random%20Forest.ipynb)
    
    
    
### Features of the Neural Network ###
1. 3 hiddenLayers. 

2. tanh Activation Function. (Also tested with ReLU and Sigmoid)

3. Xavier Initialization used. (Tested with naive strategies)

4. All fully-connected layers. 

5. L2 Norm Regularization used. 

6. RMSProp Optimizer used. (Evaluated Adam, SGD, SGD + Momentum )

7. Learning Rate of 3e-2 used. (Tried various learning rates).

8. 4-cross Validation used.


All the codes (not notebook file) can be found [here](https://github.com/leo237/activity_recognition)



