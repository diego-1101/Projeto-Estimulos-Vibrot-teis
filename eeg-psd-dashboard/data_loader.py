import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(protocol='A', filepath=None):
    """
    Loads the dataset and prepares feature matrices.
    
    Parameters:
        protocol (str): 'A' or 'B' to select which protocol data to load.
        filepath (str): Optional custom filepath. If None, uses default based on protocol.
    
    Returns:
        X_psd (pd.DataFrame): Normalized PSD features.
        X_bx (pd.DataFrame): Normalized behavioral features.
        meta (pd.DataFrame): Metadata (ID, Group, Raw Behavior).
        feature_names (dict): Dictionary of feature lists.
    """
    if filepath is None:
        filepath = f"data/df_{protocol}_final.csv"
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find data file at {filepath}")

    # --- 1. Metadata and Cleaning ---
    # Ensure ID and Group exist
    if 'ID' not in df.columns or 'grupo' not in df.columns:
        raise ValueError("Dataset missing 'ID' or 'grupo' columns.")
        
    meta_cols = ['ID', 'grupo']
    
    # --- 2. Behavioral Features ---
    # Select key behavioral metrics. Add more if available in the CSV.
    # Based on df_A_final.csv analysis: 'Desempenho', 'Acuracia', 'Similaridade', 'Especificidade'
    bx_cols = ['Desempenho', 'Acuracia', 'Similaridade', 'Especificidade']
    
    # Filter only columns that actually exist
    bx_cols = [c for c in bx_cols if c in df.columns]
    
    # --- 3. PSD Features ---
    # We want the scalar 'psd_norm_*' columns.
    # Avoiding 'psd_trecho' and 'Trecho_eeg' which are stringified arrays.
    psd_cols = [c for c in df.columns if c.startswith('psd_norm_')]
    
    if not psd_cols:
        # Fallback if norm columns aren't there, try raw 'psd_' scalar columns (checking they aren't object type)
        psd_cols = [c for c in df.columns if c.startswith('psd_') and df[c].dtype != 'O']

    # --- 4. Create Sub-DataFrames ---
    # Drop rows with NaN in critical columns (Group)
    df_clean = df.dropna(subset=['grupo']).copy()
    
    # Fill numeric NaNs with mean (simple imputation)
    # Only for feature columns
    for col in psd_cols + bx_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

    # Extract DataFrames
    meta = df_clean[meta_cols].copy()
    # Add raw behavioral data to meta for plotting/hover
    for c in bx_cols:
        meta[f'raw_{c}'] = df_clean[c]

    raw_X_psd = df_clean[psd_cols]
    raw_X_bx = df_clean[bx_cols]

    # --- 5. Normalization (Z-Score) ---
    scaler_psd = StandardScaler()
    scaler_bx = StandardScaler()

    if not raw_X_psd.empty:
        X_psd = pd.DataFrame(scaler_psd.fit_transform(raw_X_psd), columns=psd_cols, index=df_clean.index)
    else:
        X_psd = pd.DataFrame()

    if not raw_X_bx.empty:
        X_bx = pd.DataFrame(scaler_bx.fit_transform(raw_X_bx), columns=bx_cols, index=df_clean.index)
    else:
        X_bx = pd.DataFrame()

    feature_names = {
        'psd': psd_cols,
        'bx': bx_cols
    }

    return X_psd, X_bx, meta, feature_names
