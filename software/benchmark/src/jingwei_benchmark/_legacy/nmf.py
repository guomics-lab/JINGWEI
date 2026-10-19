import numpy as np
from sklearn.decomposition import NMF

def impute(full_data, full_mask):
    try:
        full_data = full_data * full_mask
        data_to_impute = full_data.copy()
        shift_value = 0.0

        observed_values = data_to_impute[full_mask == 1]
        if observed_values.size > 0:
            min_observed = np.min(observed_values)
            if min_observed < 0:
                shift_value = -min_observed + 1e-6
                data_to_impute = data_to_impute + shift_value
        data_to_impute[full_mask == 0] = 0.0
        data_to_impute = np.maximum(data_to_impute, 0.0)

        n_features = data_to_impute.shape[1]
        n_samples = data_to_impute.shape[0]
        n_components = min(n_samples -1 if n_samples > 1 else 1, n_features -1 if n_features > 1 else 1, 20)
        n_components = max(1, n_components)

        nmf = NMF(
            n_components=n_components,
            init='nndsvd',
            max_iter=100,
            random_state=42,
            tol=1e-4
        )

        W = nmf.fit_transform(data_to_impute)
        H = nmf.components_
        reconstructed_full_data = np.dot(W, H)

        if shift_value > 0:
            reconstructed_full_data = reconstructed_full_data - shift_value
            reconstructed_full_data = np.maximum(reconstructed_full_data, 0)

        final_result = full_data.copy()
        missing_mask = (full_mask == 0)
        final_result[missing_mask] = reconstructed_full_data[missing_mask]

    except Exception as e:
        print(f"    Error in NMF: {e}")

    return final_result
