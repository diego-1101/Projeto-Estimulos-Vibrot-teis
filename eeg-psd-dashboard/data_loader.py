import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(protocol='A', filepath=None):
    """
    Loads the dataset and prepares feature matrices.
    """
    # Get the directory where data_loader.py is located
    base_dir = os.path.dirname(__file__)
    
    if filepath is None:
        # Construct absolute path to data file
        filepath = os.path.join(base_dir, 'data', f'df_{protocol}_final.csv')
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        # Try checking in a few common relative locations if absolute fail
        try:
             df = pd.read_csv(f"data/df_{protocol}_final.csv")
        except:
             raise FileNotFoundError(f"Could not find data file at {filepath}")

    # Load additional metadata (Complexity, Overlap) with robust paths
    try:
        if protocol == 'A':
            comp_path = os.path.join(base_dir, 'data', "complexidade_protA.csv")
            over_path = os.path.join(base_dir, 'data', "overlap_protA.csv")
            
            comp_df = pd.read_csv(comp_path)
            over_df = pd.read_csv(over_path)
            
            if len(comp_df) == len(df):
                df['Complexidade'] = comp_df['Complexidade']
            
            if len(over_df) == len(df):
                df['Overlap'] = over_df['Overlap']
                
        elif protocol == 'B':
            comp_path = os.path.join(base_dir, 'data', "complexidade_protB.csv")
            comp_df = pd.read_csv(comp_path)
            if len(comp_df) == len(df):
                df['Complexidade'] = comp_df['Complexidade']
                
    except Exception as e:
        print(f"Warning: Could not load extra metadata: {e}")

    # --- 1. Metadata and Cleaning ---
    # Ensure ID and Group exist
    if 'ID' not in df.columns or 'grupo' not in df.columns:
        raise ValueError("Dataset missing 'ID' or 'grupo' columns.")
        
    meta_cols = ['ID', 'grupo']
    # Add new cols to meta if they exist
    if 'Complexidade' in df.columns:
        meta_cols.append('Complexidade')
    if 'Overlap' in df.columns:
        meta_cols.append('Overlap')
    
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
