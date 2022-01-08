# Multi-collinearity

Origin of the word: The word multi-collinearity consists of two words:Multi, meaning multiple, and Collinear, meaning being linearly dependent on each other.

For e.g., Let’s consider this equation a+b=1=>b=1−a
It means that ‘b’ can be represented in terms of ‘a’ i.e., if the value of ‘a’ changes, automatically the value of ‘b’ will also change. This equation denotes a simple linear relationship among two variables.

# Multi-collinearity with linear regression

Definition: The purpose of executing a Linear Regression is to predict the value of a dependent variable based on certain independent variables.

So, when we perform a Linear Regression, we want our dataset to have variables which are independent i.e., we should not be able to define an independent variable with the help of another independent variable because now in our model we have two variables which can be defined based on a certain set of independent variables which defeats the entire purpose.

Multi-collinearity is the statistical term to represent this type of a relation amongst the independent variable- when the independent variables are not so independent😊.
We can define multi-collinearity as the situation where the independent variables (or the predictors) have strong correlation amongst themselves.

The mathematical flow for multicollinearity can be shown as: 

Why Should We Care About Multi-Collinearity?
The coefficients in a Linear Regression model represent the extent of change in Y when a certain x (amongst X1,X2,X3…) is changed keeping others constant. But, if x1 and x2 are dependent, then this assumption itself is wrong that we are changing one variable keeping others constant as the dependent variable will also be changed. It means that our model itself becomes a bit flawed.
We have a redundancy in our model as two variables (or more than two) are trying to convey the same information.
As the extent of the collinearity increases, there is a chance that we might produce an overfitted model. An overfitted model works well with the test data but its accuracy fluctuates when exposed to other data sets.
Can result in a Dummy Variable Trap.
Detection
Correlation Matrices and Plots: for correlation between all the X variables.

  This plot shows the extent of correlation between the independent variable. Generally, a correlation greater than 0.9 or less than -0.9 is to be avoided.

Variance Inflation Factor: Regression of one X variable against other X variables.

VIF=1(1−Rsquared)
      The VIF factor, if greater than 10 shows extreme correlation between the variables and then we need to take care of the correlation.
Remedies for Multicollinearity
Do Nothing: If the Correlation is not that extreme, we can ignore it. If the correlated variables are not used in solving our business question, they can be ignored.
Remove One Variable: Like in dummy variable trap
Combine the correlated variables: Like creating a seniority score based on Age and Years of experience
Principal Component Analysis