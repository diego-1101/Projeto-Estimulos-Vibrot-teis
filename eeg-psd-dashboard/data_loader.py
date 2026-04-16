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
    df['Protocolo_X'] = protocol # Store protocol for downstream loaders without interfering with main names
    if 'Complexidade' in df.columns:
        meta_cols.append('Complexidade')
    if 'Overlap' in df.columns:
        meta_cols.append('Overlap')

    # Drop rows missing crucial group info (relevant for A/B, does nothing for C mostly)
    df_clean = df.dropna(subset=['grupo']).copy()
    
    meta = df_clean[meta_cols].copy()
    meta['_protocol_origin'] = protocol # Store for build_X
    
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

def build_X(df, x_mode, fase='estimulacao', selected_channels=None):
    """
    Constructs the X feature matrix based on the selected mode.
    Modes:
      - 'psd_full_norm': Concatenates arrays from selected channels.
    """
    if selected_channels is None:
         selected_channels = ['CZ', 'C3', 'C4']
         
    if x_mode == 'psd_full_norm':
        # Load pre-stacked CSV data for the specific phase
        protocol = df.get('Protocolo_X', pd.Series('A', index=df.index)).iloc[0] if 'Protocolo_X' in df.columns else 'A'
        
        base_dir = os.path.dirname(__file__)
        full_filepath = os.path.join(base_dir, 'data', f'prot{protocol}_X_psd_norm_completo_{fase}.csv')
        
        # Fallback for lowercase 'x'
        if not os.path.exists(full_filepath):
             full_filepath = os.path.join(base_dir, 'data', f'prot{protocol}_x_psd_norm_completo_{fase}.csv')
             
        try:
            full_df = pd.read_csv(full_filepath)
            # The indices must match the cleaned df
            if 'Unnamed: 0' in full_df.columns:
                full_df = full_df.drop(columns=['Unnamed: 0'])
                
            # Filter the dataframe to only keep the columns that belong to the selected channels
            # The columns are formatted as "CH_0", "CH_1", etc.
            valid_cols = [c for c in full_df.columns if c.split('_')[0] in selected_channels]
            full_df = full_df[valid_cols]
            
            if len(full_df) == len(df):
                full_df.index = df.index
                return full_df
            elif len(full_df) > len(df):
                # If df was dropped (e.g. dropna on grupo), align via inner join if possible, or naive slice
                # Since user guaranteed trials match perfectly before cleaning:
                full_df.index = df.index.parent if hasattr(df, "parent") else pd.RangeIndex(len(full_df))
                full_df_clean = full_df.loc[df.index].copy()
                return full_df_clean
            else:
                # Data is incomplete (e.g. Protocol C Stimulation has 108/144 trials).
                # Assume the PSD rows correspond to the first N trials in the metadata.
                full_df.index = df.index[:len(full_df)]
                return full_df
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing {full_filepath}. Please ensure the file is present in the data folder.")
        except Exception as e:
            raise ValueError(f"Failed loading full PSD array: {str(e)}")
    else:
        raise ValueError(f"Unknown x_mode: {x_mode} - Note that v2 requires psd_full_norm mode.")
        
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

def get_condition_n(protocol, group):
    """
    Helper to get the number of trials for a given protocol and group.
    """
    try:
        # Avoid circular imports if any, but here we are in data_loader.py
        # Protocol C handling is done in load_data
        df, _ = load_data(protocol=protocol)
        
        if protocol == 'baseline_C':
             return len(df)
             
        if group and 'grupo' in df.columns:
            n = len(df[df['grupo'] == group])
        else:
            n = len(df)
            
        return max(n, 2) # t-test needs at least 2 samples
    except Exception as e:
        print(f"Error getting n for {protocol}/{group}: {e}")
        return 30 # Default safety fallback for typical datasets
