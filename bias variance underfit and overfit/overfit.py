
# example 1 with decision tree
# evaluate decision tree performance on train and test sets with different tree depths
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot

# create dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=5, n_redundant=15, random_state=1)

# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# define lists to collect scores
train_scores, test_scores = list(), list()

# define the tree depths to evaluate
values = [i for i in range(1, 21)]

# evaluate a decision tree for each depth
for i in values:
    # configure the model
    model = DecisionTreeClassifier(max_depth=i)
    # fit model on the training dataset
    model.fit(X_train, y_train)
    # evaluate on the train dataset
    train_yhat = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_yhat)
    train_scores.append(train_acc)
    # evaluate on the test dataset
    test_yhat = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_yhat)
    test_scores.append(test_acc)
    # summarize progress
    print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))

"""
DecisionTreeClassifier(max_depth=1)
>1, train: 0.761, test: 0.769
DecisionTreeClassifier(max_depth=2)
>2, train: 0.809, test: 0.805
DecisionTreeClassifier(max_depth=3)
>3, train: 0.882, test: 0.876
DecisionTreeClassifier(max_depth=4)
>4, train: 0.901, test: 0.895
DecisionTreeClassifier(max_depth=5)
>5, train: 0.916, test: 0.904
DecisionTreeClassifier(max_depth=6)
>6, train: 0.933, test: 0.917
DecisionTreeClassifier(max_depth=7)
>7, train: 0.944, test: 0.916
DecisionTreeClassifier(max_depth=8)
>8, train: 0.953, test: 0.919
DecisionTreeClassifier(max_depth=9)
>9, train: 0.960, test: 0.922
DecisionTreeClassifier(max_depth=10)
>10, train: 0.965, test: 0.917
DecisionTreeClassifier(max_depth=11)
>11, train: 0.972, test: 0.916
DecisionTreeClassifier(max_depth=12)
>12, train: 0.979, test: 0.908
DecisionTreeClassifier(max_depth=13)
>13, train: 0.985, test: 0.907
DecisionTreeClassifier(max_depth=14)
>14, train: 0.989, test: 0.905
DecisionTreeClassifier(max_depth=15)
>15, train: 0.993, test: 0.904
DecisionTreeClassifier(max_depth=16)
>16, train: 0.995, test: 0.904
DecisionTreeClassifier(max_depth=17)
>17, train: 0.997, test: 0.903
DecisionTreeClassifier(max_depth=18)
>18, train: 0.998, test: 0.904
DecisionTreeClassifier(max_depth=19)
"""

# plot of train and test scores vs tree depth
pyplot.plot(values, train_scores, '-o', label='Train')
pyplot.plot(values, test_scores, '-o', label='Test')
pyplot.legend()
pyplot.show()

"""
- Running the example fits and evaluates a decision tree on the train and test sets for each tree depth and reports the accuracy scores.

- In this case, we can see a trend of increasing accuracy on the training dataset with the tree depth to a point around
    a depth of 19-20 levels where the tree fits the training dataset perfectly.

- We can also see that the accuracy on the test set improves with tree depth until a depth of about eight or nine levels, after
    which accuracy begins to get worse with each increase in tree depth.

- This is exactly what we would expect to see in a pattern of overfitting.

- We would choose a tree depth of eight or nine before the model begins to overfit the training dataset. """

# example 2 with KNN

# evaluate knn performance on train and test sets with different numbers of neighbors
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot

# create dataset
X, y = make_classification(n_samples=10000, n_features=20, n_informative=5, n_redundant=15, random_state=1)

# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# define lists to collect scores
train_scores, test_scores = list(), list()

# define the tree depths to evaluate
values = [i for i in range(1, 51)]
# evaluate a decision tree for each depth
for i in values:
    # configure the model
    model = KNeighborsClassifier(n_neighbors=i)
    # fit model on the training dataset
    model.fit(X_train, y_train)
    # evaluate on the train dataset
    train_yhat = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_yhat)
    train_scores.append(train_acc)
    # evaluate on the test dataset
    test_yhat = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_yhat)
    test_scores.append(test_acc)
    # summarize progress
    print('>%d, train: %.3f, test: %.3f' % (i, train_acc, test_acc))

"""
KNeighborsClassifier(n_neighbors=1)
>1, train: 1.000, test: 0.928
KNeighborsClassifier(n_neighbors=2)
>2, train: 0.963, test: 0.922
KNeighborsClassifier(n_neighbors=3)
>3, train: 0.962, test: 0.930
KNeighborsClassifier(n_neighbors=4)
>4, train: 0.956, test: 0.929
KNeighborsClassifier()
>5, train: 0.952, test: 0.931
KNeighborsClassifier(n_neighbors=6)
>6, train: 0.952, test: 0.931
KNeighborsClassifier(n_neighbors=7)
>7, train: 0.950, test: 0.933
KNeighborsClassifier(n_neighbors=8)
>8, train: 0.949, test: 0.932
KNeighborsClassifier(n_neighbors=9)
>9, train: 0.947, test: 0.930
KNeighborsClassifier(n_neighbors=10)
>10, train: 0.947, test: 0.935
KNeighborsClassifier(n_neighbors=11)
>11, train: 0.944, test: 0.933
KNeighborsClassifier(n_neighbors=12)
>12, train: 0.945, test: 0.933
KNeighborsClassifier(n_neighbors=13)
>13, train: 0.943, test: 0.931
KNeighborsClassifier(n_neighbors=14)
>14, train: 0.945, test: 0.933
KNeighborsClassifier(n_neighbors=15)
>15, train: 0.942, test: 0.932
KNeighborsClassifier(n_neighbors=16)
>16, train: 0.943, test: 0.933
KNeighborsClassifier(n_neighbors=17)
>17, train: 0.942, test: 0.934
KNeighborsClassifier(n_neighbors=18)
>18, train: 0.943, test: 0.932
KNeighborsClassifier(n_neighbors=19)
>19, train: 0.942, test: 0.931
KNeighborsClassifier(n_neighbors=20)
>20, train: 0.942, test: 0.931
KNeighborsClassifier(n_neighbors=21)
>21, train: 0.942, test: 0.931
KNeighborsClassifier(n_neighbors=22)
>22, train: 0.942, test: 0.930
KNeighborsClassifier(n_neighbors=23)
>23, train: 0.941, test: 0.930
KNeighborsClassifier(n_neighbors=24)
>24, train: 0.941, test: 0.929
KNeighborsClassifier(n_neighbors=25)
>25, train: 0.939, test: 0.930
KNeighborsClassifier(n_neighbors=26)
>26, train: 0.940, test: 0.930
KNeighborsClassifier(n_neighbors=27)
>27, train: 0.938, test: 0.930
KNeighborsClassifier(n_neighbors=28)
>28, train: 0.940, test: 0.930
KNeighborsClassifier(n_neighbors=29)
>29, train: 0.938, test: 0.930
KNeighborsClassifier(n_neighbors=30)
>30, train: 0.938, test: 0.929
KNeighborsClassifier(n_neighbors=31)
>31, train: 0.937, test: 0.930
KNeighborsClassifier(n_neighbors=32)
>32, train: 0.936, test: 0.929
KNeighborsClassifier(n_neighbors=33)
>33, train: 0.936, test: 0.929
KNeighborsClassifier(n_neighbors=34)
>34, train: 0.937, test: 0.929
KNeighborsClassifier(n_neighbors=35)
>35, train: 0.936, test: 0.928
KNeighborsClassifier(n_neighbors=36)
>36, train: 0.937, test: 0.928
KNeighborsClassifier(n_neighbors=37)
>37, train: 0.937, test: 0.928
KNeighborsClassifier(n_neighbors=38)
>38, train: 0.937, test: 0.926
KNeighborsClassifier(n_neighbors=39)
>39, train: 0.936, test: 0.926
KNeighborsClassifier(n_neighbors=40)
>40, train: 0.936, test: 0.925
KNeighborsClassifier(n_neighbors=41)
>41, train: 0.936, test: 0.927
KNeighborsClassifier(n_neighbors=42)
>42, train: 0.935, test: 0.926
KNeighborsClassifier(n_neighbors=43)
>43, train: 0.935, test: 0.927
KNeighborsClassifier(n_neighbors=44)
>44, train: 0.935, test: 0.928
KNeighborsClassifier(n_neighbors=45)
>45, train: 0.935, test: 0.928
KNeighborsClassifier(n_neighbors=46)
>46, train: 0.935, test: 0.926
KNeighborsClassifier(n_neighbors=47)
>47, train: 0.934, test: 0.927
KNeighborsClassifier(n_neighbors=48)
>48, train: 0.935, test: 0.926
KNeighborsClassifier(n_neighbors=49)
>49, train: 0.934, test: 0.926
KNeighborsClassifier(n_neighbors=50)
>50, train: 0.934, test: 0.926
 """

# plot of train and test scores vs number of neighbors
pyplot.plot(values, train_scores, '-o', label='Train')
pyplot.plot(values, test_scores, '-o', label='Test')
pyplot.legend()
pyplot.show()

"""
- Running the example fits and evaluates a KNN model on the train and test sets for each number of neighbors and reports the
    accuracy scores.

- Recall, we are looking for a pattern where performance on the test set improves and then starts to get worse, and
    performance on the training set continues to improve.

- We do not see this pattern.

- Instead, we see that accuracy on the training dataset starts at perfect accuracy and falls with almost every increase in the
    number of neighbors.

- We also see that performance of the model on the holdout test improves to a value of about five neighbors, holds level
    and begins a downward trend after that.

- A figure shows line plots of the model accuracy on the train and test sets with different numbers of neighbors.

- The plots make the situation clearer. It looks as though the line plot for the training set is dropping to converge
    with the line for the test set. Indeed, this is exactly what is happening
 """

# ref: https://machinelearningmastery.com/overfitting-machine-learning-models/