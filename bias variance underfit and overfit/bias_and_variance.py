# estimate the bias and variance for a regression model

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mlxtend.evaluate import bias_variance_decomp

# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)

# separate into inputs and outputs
data = dataframe.values
X, y = data[:, :-1], data[:, -1]

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# define the model

model = LinearRegression()
# estimate bias and variance

mse, bias, var = bias_variance_decomp(model, X_train, y_train, X_test, y_test, loss='mse', num_rounds=200, random_seed=1)

# summarize results
print('MSE: %.3f' % mse)
print('Bias: %.3f' % bias)
print('Variance: %.3f' % var)
"""
MSE: 22.418
Bias: 20.744
Variance: 1.674

- model has a high bias and a low variance.
- This is to be expected given that we are using a linear regression model.
- We can also see that the sum of the estimated mean and variance equals the estimated error of the model,
    e.g. 20.726 + 1.761 = 22.487.
"""


"""
Ref:
https://machinelearningmastery.com/calculate-the-bias-variance-trade-off/ """