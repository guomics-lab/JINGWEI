import numpy as np
from sklearn.ensemble import RandomForestRegressor

def impute(full_data, full_mask, max_iter=10, tol=1e-3, random_state=42):
    try:
        full_data = full_data.astype(float)
        mask = (full_mask == 1)
        missing_mask = (full_mask == 0)
        if not missing_mask.any():
            return full_data.copy()

        X = full_data.copy()
        for col in range(X.shape[1]):
            miss_rows = missing_mask[:, col]
            if miss_rows.any():
                obs_vals = X[mask[:, col], col]
                if obs_vals.size > 0:
                    fill_val = np.mean(obs_vals)
                else:
                    fill_val = 0.0
                X[miss_rows, col] = fill_val

        cols_with_missing = np.where(missing_mask.sum(axis=0) > 0)[0]

        col_order = cols_with_missing[np.argsort(missing_mask.sum(axis=0)[cols_with_missing])[::-1]]

        rng = np.random.RandomState(random_state)

        for it in range(max_iter):
            prev_X_missing = X[missing_mask].copy()

            for col in col_order:
                miss_rows = missing_mask[:, col]
                if not miss_rows.any():
                    continue
                obs_rows = ~miss_rows

                y_obs = X[obs_rows, col]
                X_obs = X[obs_rows, :]
                X_miss = X[miss_rows, :]
                X_obs_feat = np.delete(X_obs, col, axis=1)
                X_miss_feat = np.delete(X_miss, col, axis=1)

                if X_obs_feat.shape[1] == 0:
                    continue

                rf = RandomForestRegressor(
                    n_estimators=100,
                    random_state=rng.randint(0, 1_000_000),
                    n_jobs=-1
                )
                try:
                    rf.fit(X_obs_feat, y_obs)
                    y_pred = rf.predict(X_miss_feat)
                    X[miss_rows, col] = y_pred
                except Exception:
                    pass

            curr_X_missing = X[missing_mask]
            diff_num = np.linalg.norm(curr_X_missing - prev_X_missing)
            diff_den = np.linalg.norm(prev_X_missing) + 1e-8
            rel_change = diff_num / diff_den
            if rel_change < tol:
                break

        result = full_data.copy()
        result[missing_mask] = X[missing_mask]
        return result

    except Exception as e:
        print(f"    Error in MissForest: {e}")
        return full_data.copy()
