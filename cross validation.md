### CV

- cross-validation is a step in the process of building a machine learning model which
helps us ensure that our models fit the data accurately and also ensures that we do
not overfit.
- Cross-validation is a method of evaluating a machine learning model’s performance across random samples of the dataset.
    - This assures that any biases in the dataset are captured.
    - Cross-validation can help us to obtain reliable estimates of the model’s generalization error,    that is, how well the model performs on unseen data.
- In KNN ML model CV can help in selecting the right K value for KNN, For every k we determine test accuracy and select k that gives best accuracy on test set
    - Ex:
    1. Dataset is split randomly into train and test set;
    2. Train set is further divided randomly into k’ equal sized parts;
    3. For each k hyper parameter in kNN, we will use k’-1 parts of train set for training and the remaining 1 part of train set as cross validation data set, we then compute accuracy or model performance on cross validation data set; we will roll around the parts of train data set to get k’ accuracies for k = 1 (kNN hyperparameter), we will then average this accuracies for k = 1; as a result we will have average of k’ accuracies of the train set; this is called as k’-fold cross validation, we will repeat the k’ fold cross validation for all hyper parameter k value choices;
    4. We pick best k from best average k’ cross validation accuracies;
    5. And apply the best hyper parameter for measuring performance on test set;
- CV can be used to check whether ML model is generalizing across all folds of CV
- CV can be used to find hyperparameter values of ML model
    - ex: K in KNN

### Types of CV

- k-fold cross-validation
    - when to choose?
        - In many cases, we have to deal with small datasets and creating big validation sets
            means losing a lot of data for the model to learn.
            - In those cases, we can opt for a
                type of k-fold cross-validation where k=N, where N is the number of samples in the
                dataset. This means that in all folds of training, we will be training on all data
                samples except 1.
    - Issue
        - One should note that this type of cross-validation can be costly in terms of the time
            it takes if the model is not fast enough, but since it’s only preferable to use this
            cross-validation for small datasets, it doesn’t matter much.

- stratified k-fold cross-validation
    - When not to use k-fold / issue with k-fold ?
        - If you have a skewed dataset for binary classification with 90% positive samples and only 10%
        negative samples, you don't want to use random k-fold cross-validation.
            - Using simple k-fold cross-validation for a dataset like this can result in folds with all
                negative samples. In these cases, we prefer using stratified k-fold cross-validation.
    - Stratified k-fold cross-validation keeps the ratio of labels in each fold constant. So,
        in each fold, you will have the same 90% positive and 10% negative samples.
    - Thus, whatever metric you choose to evaluate, it will give similar results across all folds.
- hold-out based validation
    - when to choose ?
        - what should we do if we have a large amount of data? Suppose we have 1
            million samples.
            - A 5 fold cross-validation would mean training on 800k samples
            and validating on 200k.
            - Depending on which algorithm we choose, training and
            even validation can be very expensive for a dataset which is of this size.
            - In these
            cases, we can opt for a hold-out based validation.
    - Ex:
        - For a
            dataset which has 1 million samples, we can create ten folds instead of 5 and keep
            one of those folds as hold-out. This means we will have 100k samples in the holdout,
            and we will always calculate loss, accuracy and other metrics on this set and
            train on 900k samples.
            Hold-out is also used very frequently with time-series data.
- leave-one-out cross-validation
- group k-fold cross-validation