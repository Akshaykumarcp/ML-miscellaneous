In a machine learning model, there are 2 types of parameters:

- Model Parameters:
    - These are the parameters in the model that must be determined using the training data set. These are the fitted parameters.
    - Model parameters are configuration variables that are internal to the model, and a model learns them on its own.
    - For example:
        - W Weights or Coefficients of independent variables in the Linear regression model.
        - Weights or Coefficients of independent variables SVM,
        - weight, and biases of a neural network,
        - cluster centroid in clustering.
    -    y= mx+c
        - Where m is the slope of the line, and c is the intercept of the line. These two parameters are calculated by fitting the line     by minimizing RMSE, and these are known as model parameters.

    ome key points for model parameters are as follows:

    The model uses them for making predictions.
    They are learned by the model from the data itself
    These are usually not set manually.
    These are the part of the model and key to Machine Learning Algorithms.

- Hyperparameters
    - These are adjustable parameters that must be tuned in order to obtain a model with optimal performance.
    - Hyperparameters are those parameters that are explicitly defined by the user to control the learning process.

    - These are usually defined manually by the machine learning engineer.
    - One cannot know the exact best value for hyperparameters for the given problem.
    - The best value can be determined either by the rule of thumb or by trial and error.
    - Ex:-
        - K in the KNN algorithm
        - Learning rate in gradient descent
        - Number of iterations in gradient descent
        - Number of layers in a Neural Network
        - Number of neurons per layer in a Neural Network
        - Number of clusters(k) in k means clustering
        - Kernel or filter size in convolutional layers
        - Pooling size
        - Batch size
        - Choice of optimization algorithm (e.g., gradient descent, stochastic gradient descent, or Adam optimizer)
        - Choice of activation function in a neural network (nn) layer (e.g. Sigmoid, ReLU, Tanh)
        - The choice of cost or loss function the model will use

    - Ex from sklearn models
        1.    Perceptron Classifier
        Perceptron(n_iter=40, eta0=0.1, random_state=0)

        Here, n_iter is the number of iterations, eta0 is the learning rate, and random_state is the seed of the pseudo random number generator to use when shuffling the data.

        2. Train, Test Split Estimator

        train_test_split( X, y, test_size=0.4, random_state=0)

        Here, test_size represents the proportion of the dataset to include in the test split, and random_state is the seed used by the random number generator.

        3. Logistic Regression Classifier

        LogisticRegression(C=1000.0, random_state=0)

        Here, C is the inverse of regularization strength, and random_state is the seed of the pseudo random number generator to use when shuffling the data.

        4. KNN (k-Nearest Neighbors) Classifier

        KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

        Here, n_neighbors is the number of neighbors to use, p is the power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance, and euclidean_distance for p = 2.

        5. Support Vector Machine Classifier

        SVC(kernel='linear', C=1.0, random_state=0)

        Here, kernel specifies the kernel type to be used in the algorithm, for example kernel = ‘linear’, for linear classification, or kernel = ‘rbf’ for non-linear classification. C is the penalty parameter of the error term, and random_state is the seed of the pseudo random number generator used when shuffling the data for probability estimates.

        6. Decision Tree Classifier

        DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)

        Here, criterion is the function to measure the quality of a split, max_depth is the maximum depth of the tree, and random_state is the seed used by the random number generator.

        7. Lasso Regression

        Lasso(alpha = 0.1)

        Here, alpha is the regularization parameter.

        8. Principal Component Analysis

        PCA(n_components = 4)

        Here, n_components is the number of components to keep. If n_components is not set all components are kept.