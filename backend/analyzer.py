import numpy as np
from sklearn.decomposition import PCA
import warnings

def impute_em_pca(Y, max_iter=100, tol=1e-4):
    """
    Impute missing values using Expectation-Maximization (EM) algorithm for PCA.
    This effectively implements a Probabilistic PCA (PPCA) approach for missing data.
    """
    Y_filled = Y.copy()
    # Initialize missing values with column means
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_means = np.nanmean(Y_filled, axis=0)
        
    for d in range(Y.shape[1]):
        if np.isnan(col_means[d]):
            col_means[d] = 0.0
            
    mask = np.isnan(Y_filled)
    for d in range(Y.shape[1]):
        Y_filled[mask[:, d], d] = col_means[d]
        
    if not np.any(mask):
        return Y_filled

    # Determine number of components for imputation (use rank-1 for robustness, or max possible)
    n_components = min(Y.shape[1] - 1, Y.shape[0])
    n_components = max(1, n_components)
    
    for _ in range(max_iter):
        Y_old = Y_filled.copy()
        
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(Y_filled)
        Y_hat = pca.inverse_transform(scores)
        
        Y_filled[mask] = Y_hat[mask]
        
        diff = np.linalg.norm(Y_filled[mask] - Y_old[mask])
        if diff < tol:
            break
            
    return Y_filled

def run_residue_pca(residue_data, feature_names=None, calculate_csp=False, h_idx=0, n_idx=1):
    """
    Performs Probabilistic PCA (PPCA) on multi-featured residue data.
    residue_data: List of lists [residue_id, feature1, feature2, ..., featureN]
    calculate_csp: If true, calculates CSP using h_idx and n_idx.
    """
    # Find max length to pad rows that might be short due to missing trailing columns
    max_len = max(len(row) for row in residue_data)
    padded_data = []
    for row in residue_data:
        if len(row) < max_len:
            padded_data.append(row + [np.nan] * (max_len - len(row)))
        else:
            padded_data.append(row)
            
    data = np.array(padded_data, dtype=object)
    if data.ndim != 2 or data.shape[1] < 2:
        return {"error": "Insufficient data for PCA. Need at least 2 columns (ID + Feature)"}

    residue_nos = data[:, 0]
    features = data[:, 1:].astype(float) # All but first column
    n_features = features.shape[1]
    
    completely_missing_mask = np.all(np.isnan(features), axis=1)
    valid_mask = ~completely_missing_mask
    
    # Defaults for header names if not provided
    if not feature_names or len(feature_names) != n_features:
        feature_names = [f"Col {i+1}" for i in range(n_features)]
        
    csp_values = None
    if calculate_csp and n_features >= 2 and h_idx is not None and n_idx is not None:
        # Calculate CSP ignoring NaNs
        delta_h = features[:, h_idx]
        delta_n = features[:, n_idx]
        csp_values = np.sqrt(0.5 * (delta_h**2 + (0.2 * delta_n)**2))
        
    features_valid = features[valid_mask]
    
    if len(features_valid) == 0:
        return {"error": "All data is missing."}
        
    # PPCA Imputation for missing values
    features_imputed_valid = impute_em_pca(features_valid)
    
    # Standardize features (Z-score normalization)
    mean = np.mean(features_imputed_valid, axis=0)
    std = np.std(features_imputed_valid, axis=0)
    feature_means = {name: float(m) for name, m in zip(feature_names, mean)}
    feature_stds = {name: float(s) for name, s in zip(feature_names, std)}
    
    # Avoid division by zero
    std[std == 0] = 1.0
    features_scaled_valid = (features_imputed_valid - mean) / std
    
    # PCA
    n_components = min(n_features, features_scaled_valid.shape[0])
    pca = PCA(n_components=n_components)
    scores_valid = pca.fit_transform(features_scaled_valid)
    loadings = pca.components_ 
    variance_ratio = pca.explained_variance_ratio_.tolist()
    
    results = []
    
    from scipy.stats import chi2
    
    for i in range(n_components):
        variance_i = pca.explained_variance_[i]
        scores_i_valid = scores_valid[:, i]
        z_scores_sq_valid = (scores_i_valid ** 2) / variance_i if variance_i > 0 else np.zeros_like(scores_i_valid)
        pc_p_values_valid = chi2.sf(z_scores_sq_valid, df=1).tolist()
        
        scores_full = []
        p_values_full = []
        valid_idx = 0
        for is_missing in completely_missing_mask:
            if is_missing:
                scores_full.append(None)
                p_values_full.append(None)
            else:
                scores_full.append(float(scores_i_valid[valid_idx]))
                p_values_full.append(float(pc_p_values_valid[valid_idx]))
                valid_idx += 1
                
        results.append({
            "pc_index": i + 1,
            "scores": scores_full,
            "p_values": p_values_full,
            "loadings": {name: float(val) for name, val in zip(feature_names, loadings[i])},
            "success": True
        })
        
    ret = {
        "results": results,
        "variance_ratio": variance_ratio,
        "residue_nos": residue_nos.tolist(),
        "feature_names": feature_names,
        "feature_means": feature_means,
        "feature_stds": feature_stds
    }
    
    if csp_values is not None:
        ret["csp"] = np.where(np.isnan(csp_values), None, csp_values).tolist()
        ret["csp_h_col"] = feature_names[h_idx]
        ret["csp_n_col"] = feature_names[n_idx]
        
    if calculate_csp and n_features >= 2 and h_idx is not None and n_idx is not None:
        # Original std array from imputed data
        orig_std_n = std[n_idx]
        if orig_std_n != 0:
            ret["alpha_pali"] = float(std[h_idx] / orig_std_n)
        
    return ret
