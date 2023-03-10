
### Major categorical types

- Nominal variables
    - variables that have two or more categories which do not
        have any kind of order associated with them
    - Binary variables
        - Ex: gender - male and female
    - Cyclic variable
        - Monday to Sunday
- Ordinal variables
    - have “levels” or categories with a particular
        order associated with them.
    - Ex: levels - low, high and medium
    - Label Encoding
        ```
        mapping = {
        "Freezing": 0,
        "Warm": 1,
        "Cold": 2,
        "Boiling Hot": 3,
        "Hot": 4,
        "Lava Hot": 5
        }

        import pandas as pd
        df = pd.read_csv("cat_train.csv")
        df.loc[:, "ord_2"] = df.ord_2.map(mapping)

        # sklearn impl
        from sklearn import preprocessing
        # initialize LabelEncoder
        lbl_enc = preprocessing.LabelEncoder()
        # fit label encoder and transform values on ord_2 column
        # P.S: do not use this directly. fit first, then transform
        df.loc[:, "ord_2"] = lbl_enc.fit_transform(df.ord_2.values)
        ```
    - when should we do this encodings?
        - We can use this directly in many tree-based models:
        • Decision trees
        • Random forest
        • Extra Trees
        • Or any kind of boosted trees model
            o XGBoost
            o GBM
            o LightGBM
    - When we cannot do this encodings!!
        - This type of encoding cannot be used in linear models, support vector machines or
            neural networks as they expect data to be normalized (or standardized).
            For these types of models, we can binarize the data.
            - Ex:
                Freezing --> 0 --> 0 0 0
                Warm --> 1 --> 0 0 1
                Cold --> 2 --> 0 1 0
                Boiling Hot --> 3 --> 0 1 1
                Hot --> 4 --> 1 0 0
                Lava Hot --> 5 --> 1 0 1
            - if there are lot of columns store in a sparse format for saving memory
                ```
                    import numpy as np
                    from scipy import sparse
                    # create our example feature matrix
                    example = np.array(
                    [
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 0, 1]
                    ]
                    )
                    # convert numpy array to sparse CSR matrix
                    sparse_example = sparse.csr_matrix(example)
                    # print size of this sparse matrix
                    print(sparse_example.data.nbytes)
                ```
            - there is another transformation for
                categorical variables that takes even less memory. This is known as One Hot
                Encoding.
        - if there are lost of NA values, consider naming them as "rare" or "unknown" as a new category value
        - provide count of the values or count based on grouping by two columns (col2 and id)
- Binary

- refer feature engineering repo