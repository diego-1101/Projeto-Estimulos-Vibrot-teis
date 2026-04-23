import pandas as pd
import os

results = {}
for p in ['A', 'B', 'C']:
    path = f'eeg-psd-dashboard/data/analise_df_{p}_final.csv'
    if not os.path.exists(path):
        path = f'eeg-psd-dashboard/data/df_{p}_final.csv'
    
    if os.path.exists(path):
        df = pd.read_csv(path)
        p_res = {}
        if 'grupo' in df.columns:
            p_res['groups'] = df['grupo'].value_counts().to_dict()
        if 'fase' in df.columns:
            p_res['fases'] = df['fase'].value_counts().to_dict()
        else:
            p_res['total'] = len(df)
        results[p] = p_res
    else:
        results[p] = "File not found"

print(results)
