import numpy as np
from scipy.spatial.distance import cdist

from config import Generate_griddata_B, cov_to_csv, draw


def dineof_grok(background, fine_field,
                                               k_list = list(range(25, 40, 1)),
                                               tol=1e-3, max_iter=100,
                                               preserve_obs=True,
                                               cv_fraction=0.1, random_seed=42):

    def _initialize_matrix(bg, obs, res=0.125, sigma_b=2.0, L=0.05):
        X = bg.copy()
        missing = np.isnan(X)

        if not np.any(missing):
            return X, missing

        N = bg.shape[0]
        coords = np.indices((N, N)).reshape(2, -1).T * res + res / 2
        obs_flat = obs.flatten()
        mask_obs = (~np.isnan(obs_flat)) & (obs_flat != 0)
        obs_idx = np.where(mask_obs)[0]
        if obs_idx.size == 0:
            mu = np.nanmean(bg)
            X[missing] = mu
            return X, missing

        obs_pos = coords[obs_idx]
        obs_values = obs_flat[obs_idx]

        D = cdist(coords, obs_pos, metric='euclidean')

        weights = sigma_b ** 2 * np.exp(- (D ** 2) / (2 * L ** 2))

        weights_sum = np.sum(weights, axis=1, keepdims=True)
        weights_sum[weights_sum == 0] = 1  # 避免除以零
        weights_normalized = weights / weights_sum

        interpolated_values = weights_normalized @ obs_values

        X_flat = X.flatten()
        X_flat[missing.flatten()] = interpolated_values[missing.flatten()]
        X = X_flat.reshape(N, N)

        return X, missing

    def _dineof_reconstruct(X_init, mask_miss, k, tol, max_iter):
        X = X_init.copy()
        for _ in range(max_iter):
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
            Xr = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
            X_new = X.copy()
            X_new[mask_miss] = Xr[mask_miss]
            diff = np.sqrt(np.nanmean((X_new[mask_miss] - X[mask_miss]) ** 2))
            X = X_new
            if diff < tol:
                break
        return X

    def _select_best_k(bg, obs, mask_obs, k_list, cv_fraction, tol, max_iter, lambda_penalty=0.001):
        np.random.seed(42)
        indices = np.argwhere(mask_obs)
        n_cv = int(len(indices) * cv_fraction)
        cv_indices = indices[np.random.choice(len(indices), n_cv, replace=False)]

        bg_cv = bg.copy()
        cv_mask = np.zeros_like(bg, dtype=bool)
        for i, j in cv_indices:
            bg_cv[i, j] = np.nan
            cv_mask[i, j] = True

        errors = []
        adjusted_errors = []

        for k in k_list:
            X_init, mask_miss = _initialize_matrix(bg_cv, obs)
            X_rec = _dineof_reconstruct(X_init, mask_miss, k, tol, max_iter)
            pred = X_rec[cv_mask]
            true = obs[cv_mask]
            rmse = np.sqrt(np.nanmean((pred - true) ** 2))
            adjusted_rmse = rmse + lambda_penalty * k

            errors.append(rmse)
            adjusted_errors.append(adjusted_rmse)

        best_k = k_list[np.argmin(adjusted_errors)]

        print("验证 RMSE:", ["%.4f" % e for e in errors])
        print("加权后 RMSE:", ["%.4f" % ae for ae in adjusted_errors])
        print(f"带惩罚的最佳 k = {best_k}")

        return best_k

    bg = background.astype(float)
    bg[np.isclose(bg, 0)] = np.nan
    obs = fine_field.astype(float)
    obs[np.isclose(obs, 0)] = np.nan
    mask_obs = ~np.isnan(bg)

    best_k = _select_best_k(bg, obs, mask_obs, k_list, cv_fraction, tol, max_iter, lambda_penalty=0.001)


    X_init, mask_miss = _initialize_matrix(bg, obs)
    X_final = _dineof_reconstruct(X_init, mask_miss, best_k, tol, max_iter)

    if preserve_obs:
        X_final[mask_obs] = bg[mask_obs]

    return X_final

if __name__ == "__main__":
    file_path = 'dataset/2021_02_01.csv'
    file_path_AR= 'output/AR_IDW/2021_02_01.csv'
    data_preB = Generate_griddata_B(file_path_AR)
    fine_field = data_preB[:, 2].reshape(57, 57)
    data_B = Generate_griddata_B(file_path)
    background = data_B[:, 2].reshape(57, 57)

    EOF = dineof_grok(
        background=background,
        fine_field=fine_field,
    )
    cov_to_csv(EOF, 'output/GWEOF/2021_02_01.csv')
    draw(EOF, 'output/GWEOF/2021_02_01.png')


