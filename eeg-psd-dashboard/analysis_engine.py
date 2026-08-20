import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from manova1_matlab import manova1_like_matlab

def compute_embeddings(X, Y_continuous, Y_labels, method, n_components):
    """
    Computes dimensionality reduction based on selected method and input data.

    Parameters:
        X (pd.DataFrame): Primary input matrix (PSD features).
        Y_continuous (pd.DataFrame): Secondary matrix (e.g., Behavior).
        Y_labels (pd.Series/list): Group categorical labels for supervised analysis.
        method (str): 'PCA', 'LDA', 'CDA', 'PLS'.
        n_components (int): Max number of components to return.

    Returns:
        X_scores (pd.DataFrame): Component scores with columns ['C1', 'C2', ...].
        Y_scores (pd.DataFrame): Component scores for Y (when applicable).
        stats (dict): Variance explained, p-values, correlations, etc.
    """
    stats = {}
    
    X_scores = None
    Y_scores = None

    # Handle StandardScaling for PCA, LDA, PLS
    # Note: CDA centers internally in manova1_like_matlab
    if method in ['PCA', 'LDA', 'PLS']:
        # Ensure we're scaling only numeric, valid data
        scaler_x = StandardScaler()
        X_scaled = pd.DataFrame(scaler_x.fit_transform(X), columns=X.columns, index=X.index)
        
        # Scale Y if it's continuous and we need it for PCA / PLS axis mixing
        if not Y_continuous.empty:
            scaler_y = StandardScaler()
            Y_scaled = pd.DataFrame(scaler_y.fit_transform(Y_continuous), columns=Y_continuous.columns, index=Y_continuous.index)
        else:
            Y_scaled = Y_continuous
    else:
        # CDA
        X_scaled = X.copy()
        Y_scaled = Y_continuous.copy()


    # Helper to enforce num components valid ranges
    def enforce_ncomps(n_comps, df_scaled):
        if df_scaled is None or df_scaled.empty:
             return n_comps
        return min(n_comps, df_scaled.shape[0], df_scaled.shape[1])

    X_n_comps = enforce_ncomps(n_components, X_scaled)
    Y_n_comps = enforce_ncomps(n_components, Y_scaled)

    # --- PCA ---
    if method == 'PCA':
        if not X_scaled.empty:
            model_x = PCA(n_components=X_n_comps)
            x_trans = model_x.fit_transform(X_scaled)
            X_scores = pd.DataFrame(x_trans, index=X.index)
            stats['X_explained_variance'] = model_x.explained_variance_ratio_

        if not Y_scaled.empty:
            model_y = PCA(n_components=Y_n_comps)
            y_trans = model_y.fit_transform(Y_scaled)
            Y_scores = pd.DataFrame(y_trans, index=Y_continuous.index)
            stats['Y_explained_variance'] = model_y.explained_variance_ratio_
            
        # Give columns generic names C1, C2 ...
        if X_scores is not None: X_scores.columns = [f"C{i+1}" for i in range(X_scores.shape[1])]
        if Y_scores is not None: Y_scores.columns = [f"C{i+1}" for i in range(Y_scores.shape[1])]

    # --- LDA ---
    elif method == 'LDA':
        if Y_labels is None or len(np.unique(Y_labels)) < 2:
            raise ValueError("Labels with at least 2 distinct classes are required for LDA.")
        
        le = LabelEncoder()
        y_enc = le.fit_transform(Y_labels)
        n_classes = len(np.unique(y_enc))
        
        # Max n_comps for LDA is n_classes - 1 (e.g., 2 groups -> 1 dimension)
        lda_x_ncomps = min(X_n_comps, n_classes - 1)
        lda_x_ncomps = max(1, lda_x_ncomps)  # Force at least 1
        
        if not X_scaled.empty:
             model_x = LDA(n_components=lda_x_ncomps)
             x_trans = model_x.fit_transform(X_scaled, y_enc)
             X_scores = pd.DataFrame(x_trans, index=X.index)
             stats['X_explained_variance'] = model_x.explained_variance_ratio_
             
        # Also compute LDA on Y if available
        if not Y_scaled.empty:
             # Need to ensure Y features are sufficient
             lda_y_ncomps = min(Y_n_comps, n_classes - 1)
             lda_y_ncomps = max(1, lda_y_ncomps)
             
             model_y = LDA(n_components=lda_y_ncomps)
             y_trans = model_y.fit_transform(Y_scaled, y_enc)
             Y_scores = pd.DataFrame(y_trans, index=Y_continuous.index)
             stats['Y_explained_variance'] = model_y.explained_variance_ratio_
        
        if X_scores is not None: X_scores.columns = [f"C{i+1}" for i in range(X_scores.shape[1])]
        if Y_scores is not None: Y_scores.columns = [f"C{i+1}" for i in range(Y_scores.shape[1])]

    # --- PLS ---
    elif method == 'PLS':
        if Y_scaled.empty:
            raise ValueError("Behavioral data required for PLS.")
            
        pls_comps = min(X_n_comps, Y_n_comps)
        model = PLSRegression(n_components=pls_comps)
        model.fit(X_scaled, Y_scaled)
        
        X_scores = pd.DataFrame(model.x_scores_, index=X.index)
        Y_scores = pd.DataFrame(model.y_scores_, index=Y_continuous.index)
        
        X_scores.columns = [f"C{i+1}" for i in range(X_scores.shape[1])]
        Y_scores.columns = [f"C{i+1}" for i in range(Y_scores.shape[1])]
        
        corrs = [np.corrcoef(X_scores.iloc[:, i], Y_scores.iloc[:, i])[0, 1] for i in range(X_scores.shape[1])]
        stats['canonical_correlations'] = corrs

    # --- CDA (MATLAB-Like) ---
    elif method == 'CDA':
        if Y_labels is None or len(np.unique(Y_labels)) < 2:
            raise ValueError("Labels with at least 2 distinct classes are required for CDA.")
            
        if not X_scaled.empty:
            # Note: CDA uses raw unscaled data as it handles mean centering internally
            # We already passed an unscaled version to X_scaled to skip standard scaling
            d, p, cx_stats = manova1_like_matlab(X_scaled.values, Y_labels)
            
            # Extract canonical scores based on estimated components d or default fallback
            c_dims = min(n_components, cx_stats['canon'].shape[1])
            if c_dims < 1: c_dims = 1
            
            X_scores = pd.DataFrame(cx_stats['canon'][:, :c_dims], index=X.index)
            X_scores.columns = [f"C{i+1}" for i in range(X_scores.shape[1])]
            
            stats['CDA_X'] = {
                 'd': d,
                 'p': p.tolist(),
                 'lambda': cx_stats['lambda'].tolist(),
                 'chisq': cx_stats['chisq'].tolist(),
                 'eigenval': cx_stats['eigenval'][:c_dims].tolist(),
            }
            # Compute explained variance ratio from eigenvalues
            eigenvals = np.array(cx_stats['eigenval'])
            eigenvals = np.clip(eigenvals, 0, None)
            eig_sum = eigenvals.sum()
            if eig_sum > 0:
                stats['CDA_X']['explained_variance_ratio'] = (eigenvals / eig_sum).tolist()
            else:
                stats['CDA_X']['explained_variance_ratio'] = [0.0] * len(eigenvals)
            
        if not Y_scaled.empty:
            try:
                dy, py, cy_stats = manova1_like_matlab(Y_scaled.values, Y_labels)
                c_dims_y = min(n_components, cy_stats['canon'].shape[1])
                if c_dims_y < 1: c_dims_y = 1
                Y_scores = pd.DataFrame(cy_stats['canon'][:, :c_dims_y], index=Y_continuous.index)
                Y_scores.columns = [f"C{i+1}" for i in range(Y_scores.shape[1])]
                
                stats['CDA_Y'] = {
                     'd': dy,
                     'p': py.tolist(),
                     'lambda': cy_stats['lambda'].tolist(),
                     'chisq': cy_stats['chisq'].tolist(),
                     'eigenval': cy_stats['eigenval'][:c_dims_y].tolist(),
                }
                # Compute explained variance ratio for Y from eigenvalues
                eigenvals_y = np.array(cy_stats['eigenval'])
                eigenvals_y = np.clip(eigenvals_y, 0, None)
                eig_sum_y = eigenvals_y.sum()
                if eig_sum_y > 0:
                    stats['CDA_Y']['explained_variance_ratio'] = (eigenvals_y / eig_sum_y).tolist()
                else:
                    stats['CDA_Y']['explained_variance_ratio'] = [0.0] * len(eigenvals_y)
            except Exception as e:
                # E.g. singular matrix in Y subspace, which might be tiny or collinear
                print(f"Warning: CDA failed on Y space: {e}")
                
    else:
        raise ValueError(f"Method {method} not implemented.")

    return X_scores, Y_scores, stats
