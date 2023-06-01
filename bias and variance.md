#### 

- There will always be a slight difference in what our model predicts and the actual predictions. 
- These differences are called errors. 
- The goal of an analyst is not to eliminate errors but to reduce them.

- There are two types of error in machine learning. 
    - Reducible error and 
        - Bias and Variance
    - Irreducible error
        - present in a machine learning model, because of unknown variables, and whose values cannot be reduced.

- Bias and variance are the prediction errors (value predicted by model - actual value).

### Bias
- Bias is the error due to the model’s assumptions that are made to simplify it.
- For example, using simple linear regression to model the exponential growth of a virus would result in a high bias.
-  Bias is one type of error that occurs due to wrong assumptions about data such as assuming data is linear when in reality, data follows a complex function.
- Bias is simply defined as the inability of the model because of that there is some difference or error occurring between the model’s predicted value and the actual value. 
- These differences between actual or expected values and the predicted values are known as error or bias error or error due to bias. 
- The bias is an error from erroneous assumptions in the learning algorithm.
- High bias (overly simple) can cause an algorithm to miss the relevant relations between features and target outputs (underfitting).
- **Low Bias**: Low bias value means fewer assumptions are taken to build the target function. In this case, the model will closely match the training dataset.
- Low bias ML models: Decision Trees, k-Nearest Neighbors and Support Vector Machines.
- **High Bias**: High bias value means more assumptions are taken to build the target function. In this case, the model will not match the training dataset closely. 
- The high-bias model will not be able to capture the dataset trend. 
- High bias ML models: Linear Regression, Linear Discriminant Analysis and Logistic Regression.
- It is considered as the underfitting model which has a high error rate. 
- It is due to a very simplified algorithm.

For example, a linear regression model may have a high bias if the data has a non-linear relationship.

### Variance
- Variance refers to the amount that the predicted value would change if different training data was used.
- In other words, models that place a higher emphasis on the training data will have a higher variance.
- variance is the amount by which the performance of a predictive model changes when it is trained on different subsets of the training data. 
-  Variance is the variability of the model that how much it is sensitive to another subset of the training dataset. i.e how much it can adjust on the new subset of the training dataset.
- **Low variance**: Low variance means that the model is less sensitive to changes in the training data and can produce consistent estimates of the target function with different subsets of data from the same distribution. 
- This is the case of underfitting when the model fails to generalize on both training and test data.
- Low variance ML models: Linear Regression, Linear Discriminant Analysis and Logistic Regression.
- **High variance**: High variance means that the model is very sensitive to changes in the training data and can result in significant changes in the estimate of the target function when trained on different subsets of data from the same distribution. 
- This is the case of overfitting when the model performs well on the training data but poorly on new, unseen test data. 
- It fits the training data too closely that it fails on the new training dataset.
- High variance ML models: Decision Trees, k-Nearest Neighbors and Support Vector Machines.

### trade off
- If the algorithm is too simple (hypothesis with linear eq.) then it may be on high bias and low variance condition and thus is error-prone. 
- If algorithms fit too complex ( hypothesis with high degree eq.) then it may be on high variance and low bias. 
- Now the bias-variance tradeoff essentially states that there is an inverse relationship between the amount of bias and variance in a given machine learning model.

- Linear machine learning algorithms often have a high bias but a low variance.
- Nonlinear machine learning algorithms often have a low bias but a high variance.
- This means as you decrease the bias of a model, the variance increases, and vice verse.
- However, there is an optimal point in which a specific amount of bias and variance results in a minimal amount of total error (see below).
- bias-variance tradeoff is the property of a set of predictive models whereby models with a lower bias in parameter estimation have a higher variance of the parameter estimates across samples and vice versa.
- The bias-variance dilemma or problem is the conflict in trying to simultaneously minimize these two sources of error that prevent supervised learning algorithms from generalizing beyond their training set
- Example:
    - The k-nearest neighbors algorithm has low bias and high variance, but the trade-off can be changed by increasing the value of k which increases the number of neighbors that contribute t the prediction and in turn increases the bias of the model.
    - The support vector machine algorithm has low bias and high variance, but the trade-off can be changed by increasing the C parameter that influences the number of violations of the margin allowed in the training data which increases the bias but decreases the variance.
### Different Combinations of Bias-Variance

- High Bias, Low Variance: underfitting.
- High Variance, Low Bias: overfitting.
- High-Bias, High-Variance: model is not able to capture the underlying patterns in the data (high bias) and is sensitive to changes in the training data (high variance). As a result, the model will produce inconsistent and inaccurate predictions on average.
- Low Bias, Low Variance: model is able to capture the underlying patterns in the data (low bias) and is not sensitive to changes in the training data (low variance). This is the ideal scenario for a machine learning model, as it is able to generalize well to new, unseen data and produce consistent and accurate predictions. But in practice, it’s not possible.

Ref: [1](https://www.geeksforgeeks.org/bias-vs-variance-in-machine-learning/), [2](https://www.geeksforgeeks.org/ml-bias-variance-trade-off/), [3](https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/#:~:text=Bias%20is%20the%20simplifying%20assumptions,the%20bias%20and%20the%20variance.), [4](http://scott.fortmann-roe.com/docs/BiasVariance.html)