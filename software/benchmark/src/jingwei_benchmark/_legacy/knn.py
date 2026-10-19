from sklearn.impute import KNNImputer
import numpy as np

def impute(full_data, full_mask):
    data_with_nan = full_data * full_mask
    data_with_nan[full_mask == 0] = np.nan
    try:
        imputer = KNNImputer(n_neighbors=3, weights='uniform')
        imputed_full_data = imputer.fit_transform(data_with_nan)

    except Exception as e:
        print(f"    Error in KNN: {e}")

    return imputed_full_data
