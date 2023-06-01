### Overfit
- variance error, caused by model being too complex
- As we keep increasing the value of this parameter (ex: max_depth in decision tree), test accuracy remains the
same or gets worse, but the training accuracy keeps increasing.
- It means that our simple decision tree model keeps learning about the training data better and better
with an increase in max_depth, but the performance on test data does not improve
at all. This is called overfitting.
- The model fits perfectly on the training set and performs poorly when it comes to
the test set
- Training a model on a high-dimensional dataset having too many features can sometimes lead to overfitting (Fix: PCA)

### Reasons for Overfitting:
- High variance and low bias 
- The model is too complex
- The size of the training data 
- limited # of rows in dataset having large # of features

#### Solutions
- Try more data
- Regularization
    - add lambda parameter to cost function
    - dropout regularization
    - Other methods:
        - data augmentation
        - early stopping: stop training when validation set stops improving
    - ridge and lasso
- Dimentionality reduction
    - PCA
- Cross validation for ensuring model does not overfit
- NN different architecture
- feature selection: choose relevant features to decrease model complexity
- use ensemble models for reducing variance and increasing generalization
- reduce model complexity
- introduce validation set
- hyperparameter tuning

### Underfit
- bias error, caused by model being too simple

### Reasons for Underfitting:

- High bias and low variance 
- The size of the training dataset used is not enough (fewer data to build an accurate model)
- The model is too simple.
- Training data is not cleaned and also contains noise in it.
- when we try to build a linear model with fewer non-linear data

#### Solutions
- Bigger NN or use complex algo
    - If currently we're using linear regression. Try using polynomical regression.
- Increase # of features
- Try with more data
- Train longer (increase epochs)
- NN different architecture
- remove noise from data

Ref: [1](https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/), [2](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)