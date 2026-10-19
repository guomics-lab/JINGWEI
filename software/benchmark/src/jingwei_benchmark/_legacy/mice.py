# -*- coding: utf-8 -*-
"""
Multiple Imputation by Chained Equations

Original implementation from:
https://github.com/JohnnyLin12/To-Mice-or-not-to-Mice/blob/master/MICE.py

Reference:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/

Modified for AutoPIM benchmark framework.
"""

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import train_test_split

class MiceImputer(object):

    def __init__(self, seed_values = True, seed_strategy="mean", copy=True):
        self.strategy = seed_strategy
        self.seed_values = seed_values
        self.copy = copy
        self.imp = SimpleImputer(strategy=self.strategy, copy=self.copy)

    def fit_transform(self, X, method = 'Linear', iter = 5, verbose = True):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        original_null_mask = X.isna() # Store original NaN positions
        null_cols = X.columns[original_null_mask.any()].tolist()

        if self.copy:
            new_X = X.copy()
        else:
            new_X = X

        if self.seed_values:
            if verbose:
                print('Initialization of missing-values using SimpleImputer')
            imputed_data = self.imp.fit_transform(new_X)
            new_X = pd.DataFrame(imputed_data, columns=new_X.columns, index=new_X.index)
        else:
            # Fallback seeding logic (simplified, consider improving if this path is taken often)
            if verbose:
                print('Initialization of missing-values using fallback (mean of column if available)')
            for col in null_cols:
                if new_X[col].isnull().any(): # If still NaNs after potential pre-fill
                    mean_val = new_X[col].mean()
                    if pd.isna(mean_val): # If column was all NaN
                        mean_val = 0 # Default to 0 if mean is also NaN
                    new_X[col] = new_X[col].fillna(mean_val)

        model_score = {}
        MIN_SAMPLES_FOR_SPLIT = 2 # Need at least 2 samples to split

        for i in range(iter):
            if verbose:
                print('Beginning iteration ' + str(i) + ':')
            model_score[i] = []

            for column in null_cols:
                # Rows where 'column' was originally missing
                current_col_original_null_rows = original_null_mask[column]

                # Data for training the model for 'column':
                # y_train_iter: observed values of 'column'
                # X_train_iter: other columns from rows where 'column' is observed
                y_train_iter = new_X.loc[~current_col_original_null_rows, column]
                X_train_iter_all_cols = new_X.loc[~current_col_original_null_rows]

                # Ensure X_train_iter does not contain the target column 'column'
                # and also drop any columns that are entirely NaN in this subset
                X_train_iter = X_train_iter_all_cols.drop(columns=[column], errors='ignore').dropna(axis=1, how='all')

                if y_train_iter.shape[0] < MIN_SAMPLES_FOR_SPLIT or X_train_iter.empty or X_train_iter.shape[1] == 0:
                    if verbose:
                        print(f"Iter {i}, Col {column}: Skipping model training due to insufficient observed data ({y_train_iter.shape[0]} samples). Values remain from previous step or seed.")
                    model_score[i].append(np.nan) # Or some other indicator for skipped
                    continue # Skip to the next column for this iteration

                # Proceed with train_test_split as there are enough samples
                train_x, val_x, train_y, val_y = train_test_split(X_train_iter, y_train_iter, test_size=0.33, random_state=42+i)

                # Data to predict on: rows where 'column' was originally null
                # Use the same feature set as training (X_train_iter.columns)
                X_predict_iter = new_X.loc[current_col_original_null_rows, X_train_iter.columns]

                m = None
                # Determine model type based on the training target variable
                if train_y.nunique() > 2: # Regression
                    if method == 'Linear':
                        m = LinearRegression(n_jobs = -1)
                    elif method == 'Ridge':
                        m = Ridge()
                elif train_y.nunique() == 2: # Binary classification
                    if method == 'Linear': # Assuming Linear implies Logistic for binary
                        m = LogisticRegression(n_jobs = -1, solver = 'lbfgs')
                    elif method == 'Ridge': # Assuming Ridge implies RidgeClassifier for binary
                        m = RidgeClassifier()

                if m and not train_x.empty and not X_predict_iter.empty:
                    try:
                        m.fit(train_x, train_y)
                        if not val_x.empty: # Ensure val_x is not empty for scoring
                             score = m.score(val_x, val_y)
                             model_score[i].append(score)
                             if verbose:
                                 print('Iter ' + str(i) + ', Model score for ' + str(column) + ': ' + str(score))
                        else:
                            model_score[i].append(np.nan) # Cannot score if val_x is empty

                        predictions = m.predict(X_predict_iter)
                        new_X.loc[current_col_original_null_rows, column] = predictions
                    except ValueError as e:
                        if verbose:
                            print(f"Iter {i}, Col {column}: Error during model fit/predict: {e}. Values remain.")
                        model_score[i].append(np.nan) # Error occurred
                elif m and not train_x.empty and X_predict_iter.empty:
                    # Can train, but no values to predict for this column (should not happen if column is in null_cols)
                    if verbose:
                        print(f"Iter {i}, Col {column}: Model trained but no missing values to predict. This is unexpected.")
                    model_score[i].append(np.nan)
                else: # No suitable model or no data to predict
                    if verbose:
                        print(f"Iter {i}, Col {column}: No suitable model or no data to predict. Values remain.")
                    model_score[i].append(np.nan)

            valid_scores = [s for s in model_score[i] if not np.isnan(s)]
            if not valid_scores:
                model_score[i] = 0 # Or np.nan if preferred for no valid scores
            else:
                model_score[i] = sum(valid_scores) / len(valid_scores)

        return new_X

def impute(full_data, full_mask):
    try:
        data_for_imputation = full_data * full_mask
        data_for_imputation[full_mask == 0] = np.nan

        df_with_nan = pd.DataFrame(data_for_imputation)

        mice_imputer = MiceImputer(
            seed_values=True,
            seed_strategy="mean",
            copy=True
        )

        imputed_df = mice_imputer.fit_transform(
            df_with_nan,
            method='Linear',
            iter=3,
            verbose=False
        )

        predicted_data = imputed_df.to_numpy()


    except Exception as e:
        print(f"    Error in MICE: {e}")


    return predicted_data
