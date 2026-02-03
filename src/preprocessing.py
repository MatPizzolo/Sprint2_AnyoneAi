from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)


    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.
    
    binary_cat_features = working_train_df.columns[
        (working_train_df.dtypes == "object") & (working_train_df.nunique() == 2)
        #(working_train_df.dtypes == "object") & (working_train_df.nunique() <= 2)
    ].tolist()


    multicat_features = working_train_df.columns[
        (working_train_df.dtypes == "object") & (working_train_df.nunique() > 2)
    ].tolist()

    numerical_features = working_train_df.columns[
        (working_train_df.dtypes != "object")
    ].tolist()

    binary_transformer = Pipeline(steps=[
        ("ordinal", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan
        )),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler(feature_range=(0, 1)))
    ])
    
    multicat_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler(feature_range=(0, 1)))
    ])
        
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler(feature_range=(0, 1)))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("binary", binary_transformer, binary_cat_features),
            ("multicat", multicat_transformer, multicat_features),
            ("numerical", numerical_transformer, numerical_features)
        ],
        remainder="drop"  # Drop any columns not specified
    )

    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.
    preprocessor.fit(working_train_df)
    # Transform all datasets
    train = preprocessor.transform(working_train_df)
    val = preprocessor.transform(working_val_df)
    test = preprocessor.transform(working_test_df)
    print("Output train data shape: ", train.shape)
    print("Output val data shape: ", val.shape)
    print("Output test data shape: ", test.shape, "\n")
    return train, val, test


    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.


    return None
