import pandas as pd
import os

data_dir = 'eeg-psd-dashboard/data'
files = [f for f in os.listdir(data_dir) if '_X_psd_norm_completo_' in f]

for f in files:
    try:
        df = pd.read_csv(os.path.join(data_dir, f), nrows=0)
        channels = sorted(list(set(c.split('_')[0] for c in df.columns)))
        print(f"{f}: {len(channels)} channels")
        print(f"  {channels}")
    except Exception as e:
        print(f"Error reading {f}: {e}")
