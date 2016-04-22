# startupMLChallenge
Activity Recognition from Chest Mounted Accelerometer


Step 1. 

Concatenated all data from all the csv files into one pickle file.

Step 2. 

Created 5 divisions of the concatenated file. Four files for 4-cross-validation. A 5th file for testing, after I was satisfied with my cross-validation results.

Step 3. 

Visualized the data. Tried to find out what kind of classification would work best. Random Forest Classification seemed to be the best option. 

Code for random forest classification is given [here](https://github.com/leo237/startupMLChallenge/blob/master/random_forest_classification.py).


### Tested using Neural Networks

Tried testing the performance of Neural Networks as well. Build a network with 3 hidden layers on Tensorflow. 

Highlights of the networks include

1. 3 hiddenLayers. 

2. ReLU Activation Function

3. Xavier Initialization used.

4. All fully-connected layers. 

5. L2 Norm Regularization used. 

6. Adam Optimizer used. 

7. Learning Rate of 1e-4 used.

8. 4-cross Validation used.

Code for the same can be found [here](https://github.com/leo237/startupMLChallenge/blob/master/trainNew.py). In case you want to continue training from a saved model, then you can use [this](https://github.com/leo237/startupMLChallenge/blob/master/trainContinue.py)

Unfortunately, in the limited timespan, I wasn't able to train the neural network to get the best performance. The maximum that I could squeeze from the network was a 56% accuracy. 