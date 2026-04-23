import pandas as pd
import os

files = [f for f in os.listdir('eeg-psd-dashboard/data') if 'X_psd_norm_completo' in f]
for f in sorted(files):
    df = pd.read_csv(f'eeg-psd-dashboard/data/{f}')
    print(f'{f}: {len(df)}')
