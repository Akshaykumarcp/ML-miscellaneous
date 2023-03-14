### Classification Metrics

- Accuracy: (balanced dataset)
    - if you build a model that classifies 90 images accurately, your accuracy is 90% or 0.90. If
        only 83 images are classified correctly, the accuracy of your model is 83% or 0.83.
    - Issue with accuracy?
        - In case of imbalanced/skewed dataset (the number
            of samples in one class outnumber the number of samples in other class by a lot), In
            these kinds of cases, it is not advisable to use accuracy as an evaluation metric as it
            is not representative of the data.
    - Implementation
        ```
        def accuracy(y_true, y_pred):
            """
            Function to calculate accuracy
            :param y_true: list of true values
            :param y_pred: list of predicted values
            :return: accuracy score
            """
            # initialize a simple counter for correct predictions
            correct_counter = 0
            # loop over all elements of y_true
            # and y_pred "together"
            for yt, yp in zip(y_true, y_pred):
            if yt == yp:
            # if prediction is equal to truth, increase the counter
            correct_counter += 1
            # return accuracy
            # which is correct predictions over the number of samples
            return correct_counter / len(y_true)
        ```
    - better metric is precision

Before learning about precision, we need to know a few terms. Here we have
assumed that chest x-ray images with pneumothorax are positive class (1) and
without pneumothorax are negative class (0).

- True positive (TP):
    - Given an image, if your model predicts the image has
        pneumothorax, and the actual target for that image has pneumothorax, it is
        considered a true positive.
    - Outcome where the model correctly predicts the positive class.
- True negative (TN):
    - Given an image, if your model predicts that the image does not
        have pneumothorax and the actual target says that it is a non-pneumothorax image,
        it is considered a true negative.
    - Outcome where the model correctly predicts the negative class.
In simple words, if your model correctly predicts positive class, it is true positive,
and if your model accurately predicts negative class, it is a true negative.
- False positive (FP):
    - Given an image, if your model predicts pneumothorax and the
        actual target for that image is non- pneumothorax, it a false positive.
    - (Type 1 Error): Outcome where the model incorrectly predicts the positive class.
- False negative (FN):
    - Given an image, if your model predicts non-pneumothorax
        and the actual target for that image is pneumothorax, it is a false negative.
    - (Type 2 Error): Outcome where the model incorrectly predicts the negative class.

In simple words, if your model incorrectly (or falsely) predicts positive class, it is
a false positive. If your model incorrectly (or falsely) predicts negative class, it is a
false negative.
- Implementation
    ```
    def true_positive(y_true, y_pred):
        """
        Function to calculate True Positives
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: number of true positives
        """
        # initialize
        tp = 0
        for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
        tp += 1
        return tp

    def true_negative(y_true, y_pred):
        """
        Function to calculate True Negatives
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: number of true negatives
        """
        # initialize
        tn = 0
        for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
        tn += 1
        return tn

    def false_positive(y_true, y_pred):
        """
        Function to calculate False Positives
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: number of false positives
        """
        # initialize
        Approaching (Almost) Any Machine Learning Problem – Abhishek Thakur
        35
        fp = 0
        for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
        fp += 1
        return fp

    def false_negative(y_true, y_pred):
        """
        Function to calculate False Negatives
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: number of false negatives
        """
        # initialize
        fn = 0
        for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
        fn += 1
        return fn

    def accuracy_v2(y_true, y_pred):
        """
        Function to calculate accuracy using tp/tn/fp/fn
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: accuracy score
        """
        tp = true_positive(y_true, y_pred)
        fp = false_positive(y_true, y_pred)
        fn = false_negative(y_true, y_pred)
        tn = true_negative(y_true, y_pred)
        accuracy_score = (tp + tn) / (tp + tn + fp + fn)
        return accuracy_score
    ```

- Precision
    - attempts to answer “What proportion of positive identifications was actually correct?”
    - of all the predicted positives how many are actually positive
    - Precision = TP / (TP + FP)
    - Let’s say we make a new model on the new skewed dataset and our model correctly
        identified 80 non-pneumothorax out of 90 and 8 pneumothorax out of 10.
    - Thus, we identify 88 images out of 100 successfully. The accuracy is, therefore, 0.88 or 88%.
        But, out of these 100 samples, 10 non-pneumothorax images are misclassified as
        having pneumothorax and 2 pneumothorax are misclassified as not having
        pneumothorax.
        Thus, we have:
        - TP : 8
        - TN: 80
        - FP: 10
        - FN: 2
        So, our precision is 8 / (8 + 10) = 0.444. This means our model is correct 44.4%
        times when it’s trying to identify positive samples (pneumothorax).
        ```
        def precision(y_true, y_pred):
            """
            Function to calculate precision
            :param y_true: list of true values
            :param y_pred: list of predicted values
            :return: precision score
            """
            tp = true_positive(y_true, y_pred)
            fp = false_positive(y_true, y_pred)
            precision = tp / (tp + fp)
            return precision

        ```

- Recall OR True Positive Rate (TPR) OR Sensitivity
    - attempts to answer “What proportion of actual positives was identified correctly?”
    - of all actual positives how many are predicted positive
    - Recall/TPR = TP / (TP + FN)
    - In the above case recall is 8 / (8 + 2) = 0.80. This means our model identified 80%
        of positive samples correctly.
    ```
    def recall(y_true, y_pred):
        """
        Function to calculate recall
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: recall score
        """
        tp = true_positive(y_true, y_pred)
        fn = false_negative(y_true, y_pred)
        recall = tp / (tp + fn)
        return recall

    def tpr(y_true, y_pred):
        """
        Function to calculate tpr
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: tpr/recall
        """
        return recall(y_true, y_pred)
    ```

    - For a “good” model, our precision and recall values should be high.
    - We see that in
        the above example, the recall value is quite high.
    - However, precision is very low!
    - Our model produces quite a lot of false positives but less false negatives.
    - Fewer
        false negatives are good in this type of problem because you don’t want to say that
        patients do not have pneumothorax when they do.
    - That is going to be more harmful.
        But we do have a lot of false positives, and that’s not good either.

- Precision-REcall curve or issue with Precision and Recall !!
    - Usually prediction will be in probablistic manner(0.5 threshold), so based on it precision and recall varies.
    - For multiple values of threshold we can compute and plot precision-recall curve
    - post plot, it is difficult for choosing the right threshold for precision and recall
    - Both precision and recall range from 0 to 1 and a value closer to 1 is better.

- F1 score
    - F1 score is a metric that combines both precision and recall.
    - It is defined as a simple
        weighted average (harmonic mean) of precision and recall.
    - If we denote precision
        using P and recall using R, we can represent the F1 score as:
    - F1 = 2PR / (P + R) OR
    - F1 = 2TP / (2TP + FP + FN)
    - Implementation
        ```
        def f1(y_true, y_pred):
            """
            Function to calculate f1 score
            :param y_true: list of true values
            :param y_pred: list of predicted values
            :return: f1 score
            """
            p = precision(y_true, y_pred)
            r = recall(y_true, y_pred)
            score = 2 * p * r / (p + r)
            return score
        ```
    - F1 score also ranges from 0 to 1, and a perfect prediction model has an F1 of 1.
    - When dealing with datasets that have
        skewed targets, we should look at F1 (or precision and recall) instead of accuracy

- False Positive Rate FPR OR specificity OR True Negative Rate (TNR)
    - FPR = FP / (TN + FP)
    - Implementation:
    ```
    def fpr(y_true, y_pred):
        """
        Function to calculate fpr
        :param y_true: list of true values
        :param y_pred: list of predicted values
        :return: fpr
        """
        fp = false_positive(y_true, y_pred)
        tn = true_negative(y_true, y_pred)
        return fp / (tn + fp)
    ```

- Calculate TPR and FPR
    - Let’s assume that we have only 15 samples and their target values are binary:
    - Actual targets : [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    - We train a model like the random forest, and we can get the probability of when a
        sample is positive.
    - Predicted probabilities for 1: [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3,
        0.2, 0.85, 0.15, 0.99]
    - For a typical threshold of >= 0.5, we can evaluate all the above values of precision,
        recall/TPR, F1 and FPR. But we can do the same if we choose the value of the
        threshold to be 0.4 or 0.6.
    ```
    tpr_list = []
    fpr_list = []
    # actual targets
    y_true = [0, 0, 0, 0, 1, 0, 1,
    0, 0, 1, 0, 1, 0, 0, 1]
    # predicted probabilities of a sample being 1
    y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,
    0.9, 0.5, 0.3, 0.66, 0.3, 0.2,
    0.85, 0.15, 0.99]
    # handmade thresholds
    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]
    # loop over all thresholds
    for thresh in thresholds:
    # calculate predictions for a given threshold
    temp_pred = [1 if x >= thresh else 0 for x in y_pred]
    # calculate tpr
    temp_tpr = tpr(y_true, temp_pred)
    # calculate fpr
    temp_fpr = fpr(y_true, temp_pred)
    # append tpr and fpr to lists
    tpr_list.append(temp_tpr)
    fpr_list.append(temp_fpr)
    ```
    - We get tpr and fpr values for each threshold
- ROC and AUC
    - plot tpr on y axis and fpr on x-axis --> ROC (Receiver Operating Characteristic) curve
    - if we calculate the area under this ROC curve, we are calculating another metric
        which is used very often when you have a dataset which has skewed binary targets.
    - This metric is known as the Area Under ROC Curve or Area Under Curve or
        just simply AUC
    ```
    from sklearn import metrics
    In [X]: y_true = [0, 0, 0, 0, 1, 0, 1,
    ...: 0, 0, 1, 0, 1, 0, 0, 1]
    In [X]: y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,
    ...: 0.9, 0.5, 0.3, 0.66, 0.3, 0.2,
    ...: 0.85, 0.15, 0.99]
    In [X]: metrics.roc_auc_score(y_true, y_pred)
    Out[X]: 0.8300000000000001
    ```
    - AUC values range from 0 to 1.
        - AUC = 1 implies you have a perfect model. Most of the time, it means that
        you made some mistake with validation and should revisit data processing
        and validation pipeline of yours. If you didn’t make any mistakes, then
        congratulations, you have the best model one can have for the dataset you
        built it on.
        - AUC = 0 implies that your model is very bad (or very good!). Try inverting
        the probabilities for the predictions, for example, if your probability for the
        positive class is p, try substituting it with 1-p. This kind of AUC may also
        mean that there is some problem with your validation or data processing.
        - AUC = 0.5 implies that your predictions are random. So, for any binary
        classification problem, if I predict all targets as 0.5, I will get an AUC of
        0.5.
        - AUC values between 0 and 0.5 imply that your model is worse than random. Most
        of the time, it’s because you inverted the classes. If you try to invert your
        predictions, your AUC might become more than 0.5.
        - AUC values closer to 1 are
        considered good.
    - what does AUC say about our model?
        - Suppose you get an AUC of 0.85 when you build a model to detect pneumothorax
        from chest x-ray images.
        - This means that if you select a random image from your
        dataset with pneumothorax (positive sample) and another random image without
        pneumothorax (negative sample), then the pneumothorax image will rank higher
        than a non-pneumothorax image with a probability of 0.85.
    - After calculating probabilities and AUC, you would want to make predictions on
        the test set
    - Use/Benefit of ROC
        - you can use the ROC curve to choose this threshold!
        - The ROC
        curve will tell you how the threshold impacts false positive rate and true positive
            rate and thus, in turn, false positives and true positives.
        - You should choose the
            threshold that is best suited for your problem and datasets.
        - Most of the time, the top-left value on ROC curve should give you a quite good
            threshold
    - The AUC-ROC Curve is a performance measurement for classification problems that tells us how much a model is capable of distinguishing between classes.
    - A higher AUC means that a model is more accurate.

- log loss
    - Log Loss = - 1.0 * ( target * log(prediction) + (1 - target) * log(1 - prediction) )
    - Where target is either 0 or 1 and prediction is a probability of a sample belonging
        to class 1.
    - log loss penalizes quite
        high for an incorrect or a far-off prediction, i.e. log loss punishes you for being very
        sure and very wrong.
    - Implementation
        ```
        import numpy as np
        def log_loss(y_true, y_proba):
            """
            Function to calculate log loss
            :param y_true: list of true values
            :param y_proba: list of probabilities for 1
            :return: overall log loss
            """
            # define an epsilon value
            # this can also be an input
            # this value is used to clip probabilities
            epsilon = 1e-15
            # initialize empty list to store
            # individual losses
            loss = []
            # loop over all true and predicted probability values
            for yt, yp in zip(y_true, y_proba):
            # adjust probability
            # 0 gets converted to 1e-15
            # 1 gets converted to 1-1e-15
            # Why? Think about it!
            yp = np.clip(yp, epsilon, 1 - epsilon)
            # calculate loss for one sample
            temp_loss = - 1.0 * (
            yt * np.log(yp)
            + (1 - yt) * np.log(1 - yp)
            )
            # add to loss list
            loss.append(temp_loss)
            # return mean loss over all samples
            return np.mean(loss)
        ```

Note: so far we've consider for binary targets. Same way, we can apply for multi class classification

- Confusion metrics
    - A confusion matrix is nothing
        but a table of TP, FP, TN and FN
    - FP as Type-I error and FN as Type-II error.

### Multi-label classification Metrics

- Precision at k (P@k)
    - Is not same as above discussed Precision
    - If you have a list of original classes for a given
        sample and list of predicted classes for the same, precision is defined as the number
        of hits in the predicted list considering only top-k predictions, divided by k.
        If that’s confusing, it will become apparent with python code.
    ```
    def pk(y_true, y_pred, k):
        """
        This function calculates precision at k
        for a single sample
        :param y_true: list of values, actual classes
        :param y_pred: list of values, predicted classes
        :param k: the value for k
        :return: precision at a given value k
        """
        # if k is 0, return 0. we should never have this
        # as k is always >= 1
        if k == 0:
        return 0
        # we are interested only in top-k predictions
        y_pred = y_pred[:k]
        # convert predictions to set
        pred_set = set(y_pred)
        # convert actual values to set
        true_set = set(y_true)
        # find common values
        common_values = pred_set.intersection(true_set)
        # return length of common values over k
        return len(common_values) / len(y_pred[:k])
    ```
- Average precision at k (AP@k)
    - AP@k is calculated using P@k.
        For example, if we have to calculate AP@3, we calculate P@1, P@2 and P@3 and
        then divide the sum by 3.
        ```
        def apk(y_true, y_pred, k):
            """
            This function calculates average precision at k
            for a single sample
            :param y_true: list of values, actual classes
            :param y_pred: list of values, predicted classes
            :return: average precision at a given value k
            """
            # initialize p@k list of values
            pk_values = []
            # loop over all k. from 1 to k + 1
            for i in range(1, k + 1):
            # calculate p@i and append to list
            pk_values.append(pk(y_true, y_pred, i))
            # if we have no values in the list, return 0
            if len(pk_values) == 0:
            return 0
            # else, we return the sum of list over length of list
            return sum(pk_values) / len(pk_values)
        ```
- Mean average precision at k (MAP@k)
    - In machine learning,
    we are interested in all samples, and that’s why we have mean average precision
    at k or MAP@k.
    - MAP@k is just an average of AP@k and can be calculated easily
        by the following python code.
    -
        ```
        def mapk(y_true, y_pred, k):
            """
            This function calculates mean avg precision at k
            for a single sample
            :param y_true: list of values, actual classes
            :param y_pred: list of values, predicted classes
            :return: mean avg precision at a given value k
            """
            # initialize empty list for apk values
            apk_values = []
            # loop over all samples
            for i in range(len(y_true)):
            # store apk values for every sample
            apk_values.append(
            Approaching (Almost) Any Machine Learning Problem – Abhishek Thakur
            64
            apk(y_true[i], y_pred[i], k=k)
            )
            # return mean of apk values list
            return sum(apk_values) / len(apk_values)
        ```
Note: P@k, AP@k and MAP@k all range from 0 to 1 with 1 being the best.
- Log loss
    - You
        can convert the targets to binary format and then use a log loss for each column.
    - In
        the end, you can take the average of log loss in each column. This is also known as
        mean column-wise log loss.

### Regression Metrics

- Error = True Value – Predicted Value
- Absolute Error = Abs ( True Value – Predicted Value )
- mean absolute error (MAE)
    - It’s just mean of all absolute errors
    - The absolute error is the difference between the predicted values and the actual values.
    - Thus, the mean absolute error is the average of the absolute error
    ```
    import numpy as np
    def mean_absolute_error(y_true, y_pred):
        """
        This function calculates mae
        :param y_true: list of real numbers, true values
        :param y_pred: list of real numbers, predicted values
        :return: mean absolute error
        """
        # initialize error at 0
        error = 0
        # loop over all samples in the true and predicted list
        for yt, yp in zip(y_true, y_pred):
        # calculate absolute error
        # and add to error
        error += np.abs(yt - yp)
        # return mean error
        return error / len(y_true)
    ```
- squared error = ( True Value – Predicted Value )2
- mean squared error (MSE)
    - The mean squared error or MSE is similar to the MAE, except you take the average of the squared differences between the predicted values and the actual values
    ```
    def mean_squared_error(y_true, y_pred):
        """
        This function calculates mse
        :param y_true: list of real numbers, true values
        :param y_pred: list of real numbers, predicted values
        :return: mean squared error
        """
        # initialize error at 0
        error = 0
        # loop over all samples in the true and predicted list
        for yt, yp in zip(y_true, y_pred):
        # calculate squared error
        # and add to error
        error += (yt - yp) ** 2
        # return mean error
        return error / len(y_true)
    ```
- RMSE (root mean squared error)
    - RMSE = SQRT ( MSE )
- squared logarithmic error (SLE)
- mean squared logarithmic error (MSLE)
    ```
    import numpy as np
    def mean_squared_log_error(y_true, y_pred):
        """
        This function calculates msle
        :param y_true: list of real numbers, true values
        :param y_pred: list of real numbers, predicted values
        :return: mean squared logarithmic error
        """
        # initialize error at 0
        error = 0
        # loop over all samples in true and predicted list
        for yt, yp in zip(y_true, y_pred):
        # calculate squared log error
        # and add to error
        error += (np.log(1 + yt) - np.log(1 + yp)) ** 2
        # return mean error
        return error / len(y_true)
    ```
- Root mean squared logarithmic error (RMSLE)
- Percentage Error = ( ( True Value – Predicted Value ) / True Value ) * 100
- mean percentage error
    ```
    def mean_percentage_error(y_true, y_pred):
        """
        This function calculates mpe
        :param y_true: list of real numbers, true values
        :param y_pred: list of real numbers, predicted values
        :return: mean percentage error
        """
        # initialize error at 0
        error = 0
        # loop over all samples in true and predicted list
        for yt, yp in zip(y_true, y_pred):
        # calculate percentage error
        # and add to error
        error += (yt - yp) / yt
        # return mean percentage error
        return error / len(y_true)
    ```
- mean absolute percentage error (MAPE)
    ```
    def mean_abs_percentage_error(y_true, y_pred):
        """
        This function calculates MAPE
        :param y_true: list of real numbers, true values
        :param y_pred: list of real numbers, predicted values
        :return: mean absolute percentage error
        """
        # initialize error at 0
        error = 0
        # loop over all samples in true and predicted list
        for yt, yp in zip(y_true, y_pred):
        # calculate percentage error
        # and add to error
        error += np.abs(yt - yp) / yt
        # return mean percentage error
        return error / len(y_true)
    ```
- R2 (R-squared) AKA coefficient of determination
    - R-squared says how good your model fits the data.
    - R-squared
        closer to 1.0 says that the model fits the data quite well
    - whereas closer 0 means
        that model isn’t that good.
    - R-squared can also be negative when the model just
        makes absurd predictions.
    ```
    import numpy as np
    def r2(y_true, y_pred):
        """
        This function calculates r-squared score
        :param y_true: list of real numbers, true values
        :param y_pred: list of real numbers, predicted values
        :return: r2 score
        """
        # calculate the mean value of true values
        mean_true_value = np.mean(y_true)
        # initialize numerator with 0
        numerator = 0
        # initialize denominator with 0
        denominator = 0
        # loop over all true and predicted values
        for yt, yp in zip(y_true, y_pred):
        # update numerator
        numerator += (yt - yp) ** 2
        # update denominator
        denominator += (yt - mean_true_value) ** 2
        # calculate the ratio
        ratio = numerator / denominator
        # return 1 - ratio
        return 1 – ratio
    ```
- Ajusted R square
    - Every additional independent variable added to a model always increases the R² value
    - therefore, a model with several independent variables may seem to be a better fit even if it isn’t.
    - Thus, the adjusted R² compensates for each additional independent variable and only increases if each given variable improves the model above what is possible by probability.