### PCA
- Large datasets with hundreds or thousands of features often lead to redundancy especially when features are correlated with each other. - Training a model on a high-dimensional dataset having too many features can sometimes lead to overfitting (the model captures both real     and random effects).
- In addition, an overly complex model having too many features can be hard to interpret.
- One way to solve the problem of redundancy is via feature selection and dimensionality reduction techniques such as PCA.

## Principal Component Analysis (PCA)
- is a statistical method that is used for feature extraction.
- PCA is used for high-dimensional and correlated data.
- The basic idea of PCA is to transform the original space of features into the space of the principal component.
- A PCA transformation achieves the following:
    - Reduce the number of features to be used in the final model by focusing only on the components accounting for the majority of the variance in the dataset.
    - Removes the correlation between features.
- ex: https://github.com/bot13956/principal_component_analysis_iris_dataset

## Linear Discriminant Analysis (LDA)
- PCA and LDA are two data preprocessing linear transformation techniques that are often used for dimensionality reduction to select relevant features that can be used in the final machine learning algorithm.
- PCA is an unsupervised algorithm that is used for feature extraction in high-dimensional and correlated data.
- PCA achieves dimensionality reduction by transforming features into orthogonal component axes of maximum variance in a dataset.
- The goal of LDA is to find the feature subspace that optimizes class separability and reduce dimensionality
- ex: https://github.com/bot13956/linear-discriminant-analysis-iris-dataset
