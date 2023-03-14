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

#### In what scenarios overfit can occur?
- limited # of rows in dataset having large # of features

#### Solutions
- Try more data
- Regularization
    - add lambda parameter to cost function
    - dropout regularization
    - Other methods:
        - data augmentation
        - early stopping
- Dimentionality reduction
    - PCA
- Cross validation for ensuring model does not overfit
- NN different architecture

### Underfit
- bias error, caused by model being too simple


#### Solutions
- Bigger NN
- Train longer
- NN different architecture


