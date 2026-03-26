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
    
    # Removed Euclidean distance and Chi2 p-values. Outliers handled in frontend per-PC.
    
    results = []
    
    from scipy.stats import chi2
    
    for i in range(n_components):
        variance_i = pca.explained_variance_[i]
        scores_i = scores[:, i]
        z_scores_sq = (scores_i ** 2) / variance_i if variance_i > 0 else np.zeros_like(scores_i)
        pc_p_values = chi2.sf(z_scores_sq, df=1).tolist()
        
        results.append({
            "pc_index": i + 1,
            "scores": scores_i.tolist(),
            "p_values": pc_p_values,
            "loadings": {name: float(val) for name, val in zip(feature_names, loadings[i])},
            "success": True
        })
        
    ret = {
        "results": results,
        "variance_ratio": variance_ratio,
        "residue_nos": residue_nos.tolist(),
        "feature_names": feature_names
    }
    
    if csp_values is not None:
        ret["csp"] = csp_values.tolist()
        ret["csp_h_col"] = feature_names[h_idx]
        ret["csp_n_col"] = feature_names[n_idx]
        
    return ret
