import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import sobel

from config import Generate_griddata_B, cov_to_csv


def optimal_interpolation_self_grad_adapt(background, fine_field,
                                          res=0.125,
                                          sigma_b=3.0, L0=0.15, alpha=5.0,
                                          sigma_r=0.5, preserve_obs=True):
    N = background.shape[0]
    coords = np.indices((N, N)).reshape(2, -1).T * res + res/2
    bg_flat = background.flatten()
    ff_flat = fine_field.flatten()
    obs_mask = (~np.isnan(ff_flat)) & (ff_flat != 0)
    obs_idx  = np.where(obs_mask)[0]
    if obs_idx.size == 0: return background.copy()
    y_o = ff_flat[obs_idx]

    fx = sobel(background, axis=1, mode='constant') / (2*res)
    fy = sobel(background, axis=0, mode='constant') / (2*res)
    G = np.hypot(fx, fy)
    G_flat = G.flatten()

    # L_i
    L_i = L0 / (1 + alpha * G_flat)  # (N^2,)

    # B_ij
    D = cdist(coords, coords, metric='euclidean')  # (N^2, N^2)

    L_mat = np.outer(L_i, L_i)                    # (N^2, N^2)
    B = sigma_b**2 * np.exp(- (D**2) / L_mat)

    # H、R
    H = np.zeros((obs_idx.size, N*N))
    for k, idx in enumerate(obs_idx):
        H[k, idx] = 1.0
    R = np.eye(obs_idx.size) * sigma_r**2
    HBHT = H @ B @ H.T
    K = B @ H.T @ np.linalg.inv(HBHT + R)
    Hx_b = H @ bg_flat
    analysis = (bg_flat + K @ (y_o - Hx_b)).reshape(N, N)

    if preserve_obs:
        mask_bg = (~np.isnan(background)) & (background != 0)
        analysis[mask_bg] = background[mask_bg]
    return analysis


if __name__ == "__main__":
    file_path_AR = 'output/AR_IDW/2021_02_01.csv'
    file_pathB = 'dataset/2021_02_01.csv'

    data_B = Generate_griddata_B(file_pathB)
    background = data_B[:, 2].reshape(57, 57)
    data_AR = Generate_griddata_B(file_path_AR)
    background_AR = data_AR[:, 2].reshape(57, 57)

    O = optimal_interpolation_self_grad_adapt(background, background_AR)
    cov_to_csv(O, 'output/GA-OI/2021_02_01.csv')