### Overfit

- As we keep increasing the value of this parameter (ex: max_depth in decision tree), test accuracy remains the
same or gets worse, but the training accuracy keeps increasing.
- It means that our simple decision tree model keeps learning about the training data better and better
with an increase in max_depth, but the performance on test data does not improve
at all. This is called overfitting.
- The model fits perfectly on the training set and performs poorly when it comes to
the test set

