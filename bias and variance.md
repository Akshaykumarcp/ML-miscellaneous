
### Bias
- Bias is the error due to the model’s assumptions that are made to simplify it.
- For example, using simple linear regression to model the exponential growth of a virus would result in a high bias.
- The bias is an error from erroneous assumptions in the learning algorithm.
- High bias (overly simple) can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).

### Variance
- Variance refers to the amount that the predicted value would change if different training data was used.
- In other words, models that place a higher emphasis on the training data will have a higher variance.
- The bias is an error from erroneous assumptions in the learning algorithm.
- High bias (overly simple) can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).

### trade off
- Now the bias-variance tradeoff essentially states that there is an inverse relationship between the amount of bias and variance in a given machine learning model.
- This means as you decrease the bias of a model, the variance increases, and vice verse.
- However, there is an optimal point in which a specific amount of bias and variance results in a minimal amount of total error (see below).
- bias-variance tradeoff is the property of a set of predictive models whereby models with a lower bias in parameter estimation have a higher variance of the parameter estimates across samples and vice versa.
- The bias-variance dilemma or problem is the conflict in trying to simultaneously minimize these two sources of error that prevent supervised learning algorithms from generalizing beyond their training set: