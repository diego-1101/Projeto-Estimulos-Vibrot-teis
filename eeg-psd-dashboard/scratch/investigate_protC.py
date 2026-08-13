import pandas as pd
import os

data_dir = 'eeg-psd-dashboard/data'
df_c_file = os.path.join(data_dir, 'df_C_final.csv')
psd_c_file = os.path.join(data_dir, 'protC_X_psd_norm_completo_exploracao.csv')

if not os.path.exists(df_c_file):
    df_c_file = os.path.join(data_dir, 'analise_df_C_final.csv')

print(f"Checking files:")
print(f"Meta: {df_c_file}")
print(f"PSD: {psd_c_file}")

if os.path.exists(df_c_file) and os.path.exists(psd_c_file):
    df_meta = pd.read_csv(df_c_file)
    df_psd = pd.read_csv(psd_c_file)
    
    print(f"\nMetadata rows: {len(df_meta)}")
    print(f"PSD rows: {len(df_psd)}")
    
    if 'ID' in df_meta.columns:
        # Assuming ID format might be 'P01_T01' or similar
        # Let's count unique participants
        participants_meta = df_meta['ID'].apply(lambda x: str(x).split('_')[0]).unique()
        print(f"Participants in Meta ({len(participants_meta)}): {sorted(participants_meta)}")
        
        # Check if PSD file has an ID column or if we can infer it
        if 'Unnamed: 0' in df_psd.columns:
            # Maybe the unnamed column has the index/id
            # But usually build_X expects them to align or be row-by-row
            pass
            
    # Check if there are duplicate IDs in meta
    if 'ID' in df_meta.columns:
        dupes = df_meta[df_meta.duplicated('ID')]
        if not dupes.empty:
            print(f"Duplicate IDs in metadata: {len(dupes)}")

else:
    print(f"One or more files missing.")
    if not os.path.exists(df_c_file): print(f"MISSING: {df_c_file}")
    if not os.path.exists(psd_c_file): print(f"MISSING: {psd_c_file}")
