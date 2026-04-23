import pandas as pd
import os

def get_ns(p):
    path = f'eeg-psd-dashboard/data/analise_df_{p}_final.csv'
    if not os.path.exists(path):
        path = f'eeg-psd-dashboard/data/df_{p}_final.csv'
    
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n--- Protocol {p} ---")
        if 'grupo' in df.columns:
            print("Groups:\n", df['grupo'].value_counts())
        if 'fase' in df.columns:
            print("Phases:\n", df['fase'].value_counts())
        if 'grupo' in df.columns and 'fase' in df.columns:
            print("Cross-tab:\n", pd.crosstab(df['grupo'], df['fase']))
        else:
            print(f"Total: {len(df)}")

for p in ['A', 'B', 'C']:
    get_ns(p)
