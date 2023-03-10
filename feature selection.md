- remove features with very low variance
    - If the features have a very low variance (i.e. very close to 0), they
        are close to being constant and thus, do not add any value to any model at all.
    - variance also depends on scaling of the data.
    ```
    from sklearn.feature_selection import VarianceThreshold
    data = ...
    var_thresh = VarianceThreshold(threshold=0.1)
    transformed_data = var_thresh.fit_transform(data)
    # transformed data will have all columns with variance less
    # than 0.1 removed
    ```
- remove features which have a high correlation
    - correlation between different numerical features, you can use the Pearson
        correlation.
        ```
        import pandas as pd
        from sklearn.datasets import fetch_california_housing
        # fetch a regression dataset
        data = fetch_california_housing()
        X = data["data"]
        col_names = data["feature_names"]
        y = data["target"]
        # convert to pandas dataframe
        df = pd.DataFrame(X, columns=col_names)
        # introduce a highly correlated column
        df.loc[:, "MedInc_Sqrt"] = df.MedInc.apply(np.sqrt)
        # get correlation matrix (pearson)
        df.corr()
        ```
    - Ex: feature MedInc_Sqrt has a very high correlation with MedInc (0.98). We
        can thus remove one of them.

### Univariate feature selection
    - scoring of each feature against a given target.

- ANOVA F-test
- chi2

    - There are two ways of using these in scikitlearn.
        - SelectKBest: It keeps the top-k scoring features
        - SelectPercentile: It keeps the top features which are in a percentage
        specified by the user

        ```
        from sklearn.feature_selection import chi2
        from sklearn.feature_selection import f_classif
        from sklearn.feature_selection import f_regression
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import SelectPercentile
        class UnivariateFeatureSelction:
        def __init__(self, n_features, problem_type, scoring):
        """
        Custom univariate feature selection wrapper on
        different univariate feature selection models from
        scikit-learn.
        :param n_features: SelectPercentile if float else SelectKBest
        :param problem_type: classification or regression
        :param scoring: scoring function, string
        """
        # for a given problem type, there are only
        # a few valid scoring methods
        # you can extend this with your own custom
        # methods if you wish
        if problem_type == "classification":
        valid_scoring = {
        "f_classif": f_classif,
        "chi2": chi2,
        "mutual_info_classif": mutual_info_classif
        }
        else:
        valid_scoring = {
        "f_regression": f_regression,
        "mutual_info_regression": mutual_info_regression
        }
        # raise exception if we do not have a valid scoring method
        if scoring not in valid_scoring:
        raise Exception("Invalid scoring function")
        # if n_features is int, we use selectkbest
        # if n_features is float, we use selectpercentile
        # please note that it is int in both cases in sklearn
        if isinstance(n_features, int):
        self.selection = SelectKBest(
        valid_scoring[scoring],
        k=n_features
        Approaching (Almost) Any Machine Learning Problem – Abhishek Thakur
        158
        )
        elif isinstance(n_features, float):
        self.selection = SelectPercentile(
        valid_scoring[scoring],
        percentile=int(n_features * 100)
        )
        else:
        raise Exception("Invalid type of feature")
        # same fit function
        def fit(self, X, y):
        return self.selection.fit(X, y)
        # same transform function
        def transform(self, X):
        return self.selection.transform(X)
        # same fit_transform function
        def fit_transform(self, X, y):
        return self.selection.fit_transform(X, y)
        ufs = UnivariateFeatureSelction(
        n_features=0.1,
        problem_type="regression",
        scoring="f_regression"
        )
        ufs.fit(X, y)
        X_transformed = ufs.transform(X)

        ```
    - It must be noted that you can use chi2 only for data which is non-negative in nature.
    -  It’s best to create
        a wrapper for univariate feature selection that you can use for almost any new
        problem.

- greedy feature selection
    - first step is to choose a model.
    - The second step is to select a loss/scoring function.
    - third and final step is to iteratively evaluate each feature and add it to the list of “good” features if
        it improves loss/score.
    - Issues!!
        - The computational cost
            associated with this kind of method is very high.
        - It will also take a lot of time for
            this kind of feature selection to finish.
        - if you do not use this feature selection
            properly, then you might even end up overfitting the model.
    ```
    # greedy.py
    import pandas as pd
    from sklearn import linear_model
    from sklearn import metrics
    from sklearn.datasets import make_classification
    class GreedyFeatureSelection:
    """
    A simple and custom class for greedy feature selection.
    You will need to modify it quite a bit to make it suitable
    for your dataset.
    """
    def evaluate_score(self, X, y):
    """
    This function evaluates model on data and returns
    Area Under ROC Curve (AUC)
    NOTE: We fit the data and calculate AUC on same data.
    WE ARE OVERFITTING HERE.
    But this is also a way to achieve greedy selection.
    k-fold will take k times longer.
    If you want to implement it in really correct way,
    calculate OOF AUC and return mean AUC over k folds.
    This requires only a few lines of change and has been
    shown a few times in this book.
    :param X: training data
    :param y: targets
    :return: overfitted area under the roc curve
    """
    # fit the logistic regression model,
    # and calculate AUC on same data
    # again: BEWARE
    # you can choose any model that suits your data
    Approaching (Almost) Any Machine Learning Problem – Abhishek Thakur
    160
    model = linear_model.LogisticRegression()
    model.fit(X, y)
    predictions = model.predict_proba(X)[:, 1]
    auc = metrics.roc_auc_score(y, predictions)
    return auc
    def _feature_selection(self, X, y):
    """
    This function does the actual greedy selection
    :param X: data, numpy array
    :param y: targets, numpy array
    :return: (best scores, best features)
    """
    # initialize good features list
    # and best scores to keep track of both
    good_features = []
    best_scores = []
    # calculate the number of features
    num_features = X.shape[1]
    # infinite loop
    while True:
    # initialize best feature and score of this loop
    this_feature = None
    best_score = 0
    # loop over all features
    for feature in range(num_features):
    # if feature is already in good features,
    # skip this for loop
    if feature in good_features:
    continue
    # selected features are all good features till now
    # and current feature
    selected_features = good_features + [feature]
    # remove all other features from data
    xtrain = X[:, selected_features]
    # calculate the score, in our case, AUC
    score = self.evaluate_score(xtrain, y)
    # if score is greater than the best score
    # of this loop, change best score and best feature
    if score > best_score:
    this_feature = feature
    best_score = score
    # if we have selected a feature, add it
    Approaching (Almost) Any Machine Learning Problem – Abhishek Thakur
    161
    # to the good feature list and update best scores list
    if this_feature != None:
    good_features.append(this_feature)
    best_scores.append(best_score)
    # if we didnt improve during the previous round,
    # exit the while loop
    if len(best_scores) > 2:
    if best_scores[-1] < best_scores[-2]:
    break
    # return best scores and good features
    # why do we remove the last data point?
    return best_scores[:-1], good_features[:-1]
    def __call__(self, X, y):
    """
    Call function will call the class on a set of arguments
    """
    # select features, return scores and selected indices
    scores, features = self._feature_selection(X, y)
    # transform data with selected features
    return X[:, features], scores
    if __name__ == "__main__":
    # generate binary classification data
    X, y = make_classification(n_samples=1000, n_features=100)
    # transform data by greedy feature selection
    X_transformed, scores = GreedyFeatureSelection()(X, y)
    ```

- recursive feature elimination (RFE)
    - In the previous method, we started with one feature and kept adding new features, but in
        RFE, we start with all features and keep removing one feature in every iteration that
        provides the least value to a given model.
    - But how to do we know which feature offers the least value?
        - Well, if we use models like linear support vector machine
            (SVM) or logistic regression, we get a coefficient for each feature which decides
            the importance of the features.
        - In case of any tree-based models, we get feature
            importance in place of coefficients.
        - In each iteration, we can eliminate the least
            important feature and keep eliminating it until we reach the number of features
            needed. So, yes, we have the ability to decide how many features we want to keep.
    ```
    import pandas as pd
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import fetch_california_housing
    # fetch a regression dataset
    data = fetch_california_housing()
    X = data["data"]
    col_names = data["feature_names"]
    y = data["target"]
    # initialize the model
    model = LinearRegression()
    # initialize RFE
    rfe = RFE(
    estimator=model,
    n_features_to_select=3
    )
    # fit RFE
    rfe.fit(X, y)
    # get the transformed data with
    # selected columns
    X_transformed = rfe.transform(X)
    ```

- coefficients/importance threshold
    - fit the model to the data and select features from the model by the feature
        coefficients or the importance of features.
    - If you use coefficients, you can select
        a threshold, and if the coefficient is above that threshold, you can keep the feature
        else eliminate it.

        ```
        import pandas as pd
        from sklearn.datasets import load_diabetes
        from sklearn.ensemble import RandomForestRegressor
        # fetch a regression dataset
        # in diabetes data we predict diabetes progression
        # after one year based on some features
        data = load_diabetes()
        X = data["data"]
        col_names = data["feature_names"]
        y = data["target"]
        # initialize the model
        model = RandomForestRegressor()
        # fit the model
        model.fit(X, y)
        # Feature importance from random forest (or any model) can be plotted as follows.
        importances = model.feature_importances_
        idxs = np.argsort(importances)
        plt.title('Feature Importances')
        plt.barh(range(len(idxs)), importances[idxs], align='center')
        plt.yticks(range(len(idxs)), [col_names[i] for i in idxs])
        plt.xlabel('Random Forest Feature Importance')
        plt.show()
        ```
    - You can choose
        features from one model and use another model to train.
    - For example, you can use
        Logistic Regression coefficients to select the features and then use Random Forest
        to train the model on chosen features.
    - Scikit-learn also offers SelectFromModel
        class that helps you choose features directly from a given model.

        ```
        import pandas as pd
        from sklearn.datasets import load_diabetes
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import SelectFromModel
        # fetch a regression dataset
        # in diabetes data we predict diabetes progression
        # after one year based on some features
        data = load_diabetes()
        X = data["data"]
        col_names = data["feature_names"]
        y = data["target"]
        # initialize the model
        model = RandomForestRegressor()
        # select from the model
        sfm = SelectFromModel(estimator=model)
        X_transformed = sfm.fit_transform(X, y)
        # see which features were selected
        support = sfm.get_support()
        # get feature names
        print([
        x for x, y in zip(col_names, support) if y == True
        ])
        ```

- L1 (Lasso) penalization.
    - When we have L1
        penalization for regularization, most coefficients will be 0 (or close to 0), and we
        select the features with non-zero coefficients.
    - You can do it by just replacing  random forest in the snippet of selection from a model with a model that supports
        L1 penalty, e.g. lasso regression.
    - All tree based models provide feature importance
        so all the model-based snippets can be used for XGBoost,
        LightGBM or CatBoost.
    - The feature importance function names might be different
    and may produce results in a different format, but the usage will remain the same.
    - In the end, you must be careful when doing feature selection.
    - Select features on
    training data and validate the model on validation data for proper selection of
    features without overfitting the model.
