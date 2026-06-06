import numpy as np
from scipy.stats import skew, kurtosis

def extract_paper_token_features(token_ces):
    """
    PAPER Section 3.2: Extract 5 statistical features from token CEs
    """
    token_ces = np.array(token_ces)
    
    if len(token_ces) == 0:
        return np.zeros(5)
    
    return np.array([
        np.mean(token_ces),
        np.std(token_ces),
        np.max(token_ces) - np.min(token_ces),
        skew(token_ces) if len(token_ces) > 2 else 0.0,
        kurtosis(token_ces) if len(token_ces) > 3 else 0.0
    ])


def build_features(token_ces, layer_ces):
    """
    Build combined feature vector:
    - Token: 5 statistical features (PAPER method)
    - Layer: raw causal effects (can be from skip or concatenated from multiple strategies)
    """
    token_feats = extract_paper_token_features(token_ces)
    layer_feats = np.array(layer_ces)
    
    return np.concatenate([token_feats, layer_feats])


def feature_names(num_layers):
    """Return canonical feature names for the combined vector"""
    token_names = ['token_mean', 'token_std', 'token_range', 'token_skew', 'token_kurt']
    layer_names = [f'layer_ce_{i}' for i in range(num_layers)]
    return token_names + layer_names


def adapt_features_to_model(feat_vec, model, fallback_fill=0.0):
    """Adapt feature vector to model's expected dimensionality"""
    feat = np.asarray(feat_vec, dtype=float).ravel()
    model_feat_count = None
    
    try:
        if hasattr(model, 'n_features_in_'):
            model_feat_count = int(model.n_features_in_)
        elif hasattr(model, 'coef_'):
            coef = getattr(model, 'coef_')
            if hasattr(coef, 'shape'):
                model_feat_count = int(coef.shape[-1])
    except Exception:
        model_feat_count = None

    if model_feat_count is None:
        return feat, None

    if model_feat_count == feat.size:
        return feat, model_feat_count

    if model_feat_count < feat.size:
        return feat[:model_feat_count], model_feat_count

    pad = np.full((model_feat_count - feat.size,), float(fallback_fill), dtype=float)
    return np.concatenate([feat, pad]), model_feat_count