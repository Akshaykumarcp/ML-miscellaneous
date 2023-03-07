### CV

- cross-validation is a step in the process of building a machine learning model which
helps us ensure that our models fit the data accurately and also ensures that we do
not overfit.

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