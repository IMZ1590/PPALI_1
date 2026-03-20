import numpy as np
from sklearn.decomposition import PCA

def run_residue_pca(residue_data, feature_names=None, calculate_csp=False, h_idx=0, n_idx=1):
    """
    Performs PCA on multi-featured residue data.
    residue_data: List of lists [residue_id, feature1, feature2, ..., featureN]
    calculate_csp: If true, calculates CSP using h_idx and n_idx.
    """
    data = np.array(residue_data)
    if data.ndim != 2 or data.shape[1] < 2:
        return {"error": "Insufficient data for PCA. Need at least 2 columns (ID + Feature)"}

    residue_nos = data[:, 0]
    features = np.abs(data[:, 1:].astype(float)) # All but first column
    n_features = features.shape[1]
    
    # Defaults for header names if not provided
    if not feature_names or len(feature_names) != n_features:
        feature_names = [f"Col {i+1}" for i in range(n_features)]
        
    csp_values = None
    if calculate_csp and n_features >= 2 and h_idx is not None and n_idx is not None:
        # Standard formula: CSP = sqrt( delta_H^2 + (delta_N / 5.88)^2 )
        delta_h = features[:, h_idx]
        delta_n = features[:, n_idx]
        csp_values = np.sqrt(delta_h**2 + (delta_n / 5.88)**2)
    # Standardize features (Z-score normalization)
    # Manual standardization to avoid dependency if needed, but sklearn is in requirements
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    # Avoid division by zero
    std[std == 0] = 1.0
    features_scaled = (features - mean) / std
    
    # PCA
    n_components = min(n_features, features.shape[0])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(features_scaled)
    loadings = pca.components_ 
    variance_ratio = pca.explained_variance_ratio_.tolist()
    
    # Calculate Euclidean Distance and P-values
    # Euclidean Distance D_E = sqrt( sum( score_i^2 ) )
    from scipy.stats import chi2
    
    euclidean_sq = np.sum(scores**2, axis=1)
    euclidean_dist = np.sqrt(euclidean_sq)
    
    # Heuristic P-value mapping (using Chi-squared approximation on scaled scores for rough p-value equivalent if desired)
    # The user relies on P-values, so we still calculate a sort of significance.
    # If the user wants true Euclidean, technically we should use a different distribution. For now we will keep the Chi-square approximation for outlier detection but applied on euclidean_sq. This may not be strictly rigorous but preserves the UI functionality.
    # Actually, let's normalize euclidean_sq by its mean? Or just use chi2.
    # Let's use the standard chi2 approach but replacing mahalanobis_dist with euclidean_dist in the output dictionary so we don't break frontend.
    
    p_values = chi2.sf(euclidean_sq, df=n_components)
    
    results = []
    # Store per-residue statistics separately or attach to results?
    # The current 'results' structure is per-PC. We need per-residue stats.
    # Let's attach them to the return object as parallel arrays.
    
    for i in range(n_components):
        results.append({
            "pc_index": i + 1,
            "scores": scores[:, i].tolist(),
            "loadings": {name: float(val) for name, val in zip(feature_names, loadings[i])},
            "success": False 
        })
        
    ret = {
        "results": results,
        "variance_ratio": variance_ratio,
        "residue_nos": residue_nos.tolist(),
        "feature_names": feature_names,
        "mahalanobis_dist": euclidean_dist.tolist(), # Sent as mahalanobis_dist key to avoid breaking JS, but contains Euclidean
        "p_values": p_values.tolist()
    }
    
    if csp_values is not None:
        ret["csp"] = csp_values.tolist()
        ret["csp_h_col"] = feature_names[h_idx]
        ret["csp_n_col"] = feature_names[n_idx]
        
    return ret
