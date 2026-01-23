import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.preprocessing import LabelEncoder

def compute_embedding(X, Y_labels=None, Y_continuous=None, method='PCA', covariance_mode='auto', n_components=2):
    """
    Computes dimensionality reduction based on selected method and input data.

    Parameters:
        X (pd.DataFrame): Primary input matrix (e.g., PSD features).
        Y_labels (pd.Series): Group labels (CV/SV) for discriminant analysis.
        Y_continuous (pd.DataFrame): Secondary matrix for Cross-Covariance (e.g., Behavior).
        method (str): 'PCA', 'LDA', 'CDA', 'PLS'.
        covariance_mode (str): 'auto' (Auto-covariance) or 'cross' (Cross-covariance).
        n_components (int): Number of components to return.

    Returns:
        embedding_df (pd.DataFrame): Component scores with columns ['C1', 'C2', ...].
        stats (dict): Variance explained, p-values, or correlations.
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    # Cap components at min(n_samples, n_features)
    n_comps = min(n_components, n_samples, n_features)
    
    model = None
    transformed_data = None
    stats = {}

    # --- 1. PCA (Auto-covariance only) ---
    if method == 'PCA':
        n_comps = min(n_comps, n_features)
        model = PCA(n_components=n_comps)
        transformed_data = model.fit_transform(X)
        stats['explained_variance'] = model.explained_variance_ratio_

    # --- 2. LDA (Discriminant Analysis - Supervised) ---
    # Used for "Behavior-only CDA" or "PSD-only CDA" if strictly separating groups.
    elif method == 'LDA' or (method == 'CDA' and covariance_mode == 'auto'):
        if Y_labels is None:
            raise ValueError("Labels are required for LDA/CDA (Auto-covariance).")
        
        # LDA max components is min(n_classes - 1, n_features)
        # For 2 groups (CV/SV), we only get 1 component.
        # To show a 2D plot, we might need a trick or just show 1D.
        # Alternatively, scientific "CDA" often refers to subspace projection.
        # Here we will use sklearn's LDA. 
        
        # Encoder labels to integers
        le = LabelEncoder()
        y_enc = le.fit_transform(Y_labels)
        n_classes = len(np.unique(y_enc))
        
        lda_n_comps = min(n_comps, n_classes - 1)
        if lda_n_comps < 1: lda_n_comps = 1 # Force at least 1 for valid call, though it might fail if 1 class
        
        model = LDA(n_components=lda_n_comps)
        X_trans = model.fit_transform(X, y_enc)
        
        # If we requested 2D but LDA only gives 1D (2 groups), we pad with zeros or noise for visualization
        if X_trans.shape[1] < n_components:
            # Pad with zeros for visualization
            pad_width = n_components - X_trans.shape[1]
            transformed_data = np.hstack([X_trans, np.zeros((n_samples, pad_width))])
        else:
            transformed_data = X_trans
            
        stats['explained_variance'] = model.explained_variance_ratio_

    # --- 3. PLS (Partial Least Squares) ---
    elif method == 'PLS':
        if covariance_mode == 'cross':
            # Mode A: PSD x Behavior
            if Y_continuous is None:
                raise ValueError("Behavioral data required for PLS Cross-Covariance.")
            
            # Use PLSRegression to find relations between Brain (X) and Behavior (Y)
            model = PLSRegression(n_components=n_comps)
            model.fit(X, Y_continuous)
            transformed_data = model.x_scores_ # Latent variable of X (Brain)
            
            # Calculate correlation between Latent X and Latent Y
            x_scores = model.x_scores_
            y_scores = model.y_scores_
            corrs = [np.corrcoef(x_scores[:, i], y_scores[:, i])[0, 1] for i in range(x_scores.shape[1])]
            stats['canonical_correlations'] = corrs

        elif covariance_mode == 'auto':
            # Mode B: PLS-DA (PSD x Groups)
            # Encode Y as dummy matrix (One-Hot) for PLS Discriminant Analysis
            if Y_labels is None:
                raise ValueError("Group labels required for PLS-DA.")
            
            y_dummies = pd.get_dummies(Y_labels)
            
            model = PLSRegression(n_components=n_comps)
            model.fit(X, y_dummies)
            transformed_data = model.x_scores_
            stats['note'] = "PLS-DA (Discriminant Analysis)"

    # --- 4. CDA (Canonical Correlation Analysis as Cross-Covariance) ---
    elif method == 'CDA' and covariance_mode == 'cross':
        # Scientific context: Canonical Correlation Analysis (CCA) between two mats
        if Y_continuous is None:
            raise ValueError("Behavioral data required for Canonical Correlation.")
            
        model = PLSCanonical(n_components=n_comps, algorithm='nipals')
        model.fit(X, Y_continuous)
        
        # PLSCanonical uses transform() method, not x_scores_ attribute
        x_scores, y_scores = model.transform(X, Y_continuous)
        transformed_data = x_scores
        
        # Calculate canonical correlations
        corrs = [np.corrcoef(x_scores[:, i], y_scores[:, i])[0, 1] for i in range(x_scores.shape[1])]
        stats['canonical_correlations'] = corrs

    else:
        raise ValueError(f"Method {method} with mode {covariance_mode} not implemented.")

    # --- Format Output ---
    # Create DataFrame for plotting
    cols = [f'C{i+1}' for i in range(n_components)]
    # Ensure transformed data has enough columns (pad if needed, already handled for LDA)
    if transformed_data.shape[1] < n_components:
         pad_width = n_components - transformed_data.shape[1]
         transformed_data = np.hstack([transformed_data, np.zeros((n_samples, pad_width))])
         
    embedding_df = pd.DataFrame(transformed_data[:, :n_components], columns=cols, index=X.index)
    
    return embedding_df, stats
