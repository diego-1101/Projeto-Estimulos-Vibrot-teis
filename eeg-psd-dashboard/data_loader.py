import pandas as pd
import numpy as np
import os
import ast
from sklearn.preprocessing import StandardScaler

def load_data(protocol='A', filepath=None):
    """
    Loads the dataset and prepares basic metadata.
    Handles 'analise_df_{protocol}_final.csv' with a fallback to 'df_{protocol}_final.csv'.
    """
    base_dir = os.path.dirname(__file__)
    
    if filepath is None:
        filepath = os.path.join(base_dir, 'data', f'analise_df_{protocol}_final.csv')
        if not os.path.exists(filepath):
            filepath = os.path.join(base_dir, 'data', f'df_{protocol}_final.csv')
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        try:
             df = pd.read_csv(f"data/analise_df_{protocol}_final.csv")
        except FileNotFoundError:
             try:
                 df = pd.read_csv(f"data/df_{protocol}_final.csv")
             except:
                 raise FileNotFoundError(f"Could not find data file at {filepath}")

    # --- 1. Protocol C Group Handling ---
    if protocol == 'C' or 'grupo' not in df.columns:
        df['grupo'] = 'ALL'

    # Load additional metadata (Complexity, Overlap) with robust paths if not already in df
    try:
        if protocol == 'A':
            if 'Complexidade' not in df.columns:
                comp_path = os.path.join(base_dir, 'data', "complexidade_protA.csv")
                comp_df = pd.read_csv(comp_path)
                if len(comp_df) == len(df):
                    df['Complexidade'] = comp_df['Complexidade']
            if 'Overlap' not in df.columns:
                over_path = os.path.join(base_dir, 'data', "overlap_protA.csv")
                over_df = pd.read_csv(over_path)
                if len(over_df) == len(df):
                    df['Overlap'] = over_df['Overlap']
                
        elif protocol == 'B':
            if 'Complexidade' not in df.columns:
                comp_path = os.path.join(base_dir, 'data', "complexidade_protB.csv")
                comp_df = pd.read_csv(comp_path)
                if len(comp_df) == len(df):
                    df['Complexidade'] = comp_df['Complexidade']
    except Exception as e:
        print(f"Warning: Could not load extra metadata: {e}")

    # Ensure ID exists
    if 'ID' not in df.columns:
        raise ValueError("Dataset missing 'ID' column.")
        
    # Build meta
    meta_cols = ['ID', 'grupo']
    if 'Complexidade' in df.columns:
        meta_cols.append('Complexidade')
    if 'Overlap' in df.columns:
        meta_cols.append('Overlap')

    # Drop rows missing crucial group info (relevant for A/B, does nothing for C mostly)
    df_clean = df.dropna(subset=['grupo']).copy()
    
    meta = df_clean[meta_cols].copy()
    
    # Store raw behavioral cols for hover text safely
    possible_bx = ['Desempenho', 'Acuracia', 'Similaridade', 'Especificidade', 'Proporção espacial x', 'Proporção espacial y', 'Proporção Espacial x', 'Proporção Espacial y']
    for c in possible_bx:
        if c in df_clean.columns:
            meta[f'raw_{c}'] = df_clean[c].fillna("N/A")

    return df_clean, meta

def _parse_eeg_array(val):
    if pd.isna(val):
        return np.array([])
    if isinstance(val, str):
        # Handle formats like "[1, 2, 3]" or "1 2 3" or raw lists
        val = val.strip()
        if val.startswith('['):
            try:
                # ast.literal_eval safely parses Python literals
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                     return np.array(parsed)
            except:
                pass
        # Fallback for space separated numeric strings
        val = val.replace('[', '').replace(']', '').replace(',', ' ')
        try:
            return np.array([float(x) for x in val.split()])
        except:
            return np.array([])
    if isinstance(val, (list, np.ndarray)):
        return np.array(val)
    return np.array([])

def build_X(df, x_mode):
    """
    Constructs the X feature matrix based on the selected mode.
    Modes:
      - 'psd_full': Concatenates arrays from CZ, C3, C4.
      - 'psd_bands': specific scalar unnormalized band columns.
      - 'psd_bands_norm': specific scalar normalized band columns.
    """
    if x_mode == 'psd_full':
        required_cols = ['CZ', 'C3', 'C4']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for full PSD: {missing}")
            
        full_features = []
        for idx, row in df.iterrows():
            cz = _parse_eeg_array(row['CZ'])
            c3 = _parse_eeg_array(row['C3'])
            c4 = _parse_eeg_array(row['C4'])
            # Ensure lengths match in a rough way, pad with 0 if necessary
            concated = np.concatenate([cz, c3, c4]) if (len(cz) and len(c3) and len(c4)) else np.array([])
            full_features.append(concated)
            
        # Standardize lengths inside the batch
        max_len = max([len(f) for f in full_features] + [0])
        if max_len == 0:
            return pd.DataFrame()
            
        # Ensure uniform lengths by zero padding any array that is short
        padded = []
        for f in full_features:
            if len(f) < max_len:
                padded.append(np.pad(f, (0, max_len - len(f))))
            else:
                padded.append(f)

        feat_names = [f"F_{i}" for i in range(max_len)] # Generic names
        
        X = pd.DataFrame(padded, columns=feat_names, index=df.index)
        
    elif x_mode == 'psd_bands' or x_mode == 'psd_bands_norm':
        prefix = 'psd_norm_' if x_mode == 'psd_bands_norm' else 'psd_'
        bands = ['delta', 'theta', 'alfa', 'beta', 'gamma']
        channels = ['CZ', 'C3', 'C4']
        
        target_cols = [f"{prefix}{b}_{c}" for c in channels for b in bands]
        
        # In Prot C, alfa might be spelled alpha, double check or just use what exists:
        available_cols = [c for c in target_cols if c in df.columns]

        if len(available_cols) < len(target_cols):
             target_cols_alt = [c.replace('alfa', 'alpha') for c in target_cols]
             available_cols = [c for c in target_cols_alt if c in df.columns]
             
        if not available_cols:
             # Just matching prefix
             available_cols = [c for c in df.columns if c.startswith(prefix) and not c.startswith("psd_trecho")]

        if not available_cols:
             raise ValueError(f"No columns matching {prefix} pattern found.")
             
        X = df[available_cols].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X.fillna(X.mean(), inplace=True)
        
    else:
        raise ValueError(f"Unknown x_mode: {x_mode}")
        
    return X


def build_Y(df, y_cols_selected):
    """
    Constructs the Y feature matrix based on the selected checkboxes.
    """
    if not y_cols_selected:
        raise ValueError("Select at least one Y variable.")
    
    valid_cols = []
    # Try literal match, then try lower, etc.
    for col in y_cols_selected:
        if col in df.columns:
            valid_cols.append(col)
        elif col == 'Proporção Espacial x' and 'Proporção espacial x' in df.columns:
            valid_cols.append('Proporção espacial x')
        elif col == 'Proporção Espacial y' and 'Proporção espacial y' in df.columns:
             valid_cols.append('Proporção espacial y')
        else:
            print(f"Warning: Requested Y column not found: {col}")
            
    if not valid_cols:
         raise ValueError(f"None of the selected Y variables exist in the dataset: {y_cols_selected}")
         
    Y = df[valid_cols].copy()
    for col in Y.columns:
        Y[col] = pd.to_numeric(Y[col], errors='coerce')
    Y.fillna(Y.mean(), inplace=True)
    
    return Y
