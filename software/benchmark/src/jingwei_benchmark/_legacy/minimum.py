import numpy as np

def impute(full_data, full_mask):
    data_with_nan = full_data * full_mask
    data_with_nan[full_mask == 0] = np.nan
    column_mins_np_array = np.zeros(data_with_nan.shape[1])

    for col_idx in range(data_with_nan.shape[1]):
        col_data = data_with_nan[:, col_idx]
        non_zero_non_nan = col_data[(~np.isnan(col_data)) & (col_data != 0)]
        column_mins_np_array[col_idx] = np.min(non_zero_non_nan)


    imputed_full_data = data_with_nan.copy()
    nan_indices = np.where(np.isnan(imputed_full_data))
    imputed_full_data[nan_indices] = np.take(column_mins_np_array, nan_indices[1])

    return imputed_full_data
