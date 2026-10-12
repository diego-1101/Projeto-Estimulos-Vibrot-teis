import pandas as pd
import os

data_dir = r"eeg-psd-dashboard/data"

def get_suj_col(df):
    for c in ['sujeito', 'Sujeito', 'subject', 'Subject', 'ID', 'id']:
        if c in df.columns:
            return c
    return None

def get_grp_col(df):
    for c in ['grupo', 'Grupo', 'group', 'Group']:
        if c in df.columns:
            return c
    return None

print("="*60)
print("1. PERFORMANCE DATASETS (df_prot*_performance.csv)")
print("="*60)
for prot in ['A', 'B', 'C']:
    f = os.path.join(data_dir, f"df_prot{prot}_performance.csv")
    df = pd.read_csv(f)
    print(f"\n--- Protocol {prot} (Performance) ---")
    print("Columns:", list(df.columns)[:10], "...")
    print("Total rows (trials):", len(df))
    scol = get_suj_col(df)
    gcol = get_grp_col(df)
    if scol:
        sujs = sorted(df[scol].unique())
        print(f"Subjects ({scol}) (Total N={len(sujs)}): {sujs}")
        if gcol:
            for g in sorted(df[gcol].unique()):
                g_sujs = sorted(df[df[gcol] == g][scol].unique())
                trials_count = len(df[df[gcol] == g])
                print(f"  Group {g} (N={len(g_sujs)}): {g_sujs} -> trials: {trials_count}")
    else:
        print("No subject column found!")
    if 'Fase' in df.columns:
        print("  Fases:", df['Fase'].value_counts().to_dict())

print("\n" + "="*60)
print("2. MULTIVARIATE / EEG METADATA (analise_df_*_final.csv)")
print("="*60)
for prot in ['A', 'B', 'C']:
    f = os.path.join(data_dir, f"analise_df_{prot}_final.csv")
    df = pd.read_csv(f)
    print(f"\n--- Protocol {prot} (analise_df_final / EEG Multivariate) ---")
    print("Columns:", list(df.columns)[:10], "...")
    print("Total rows (trials):", len(df))
    scol = get_suj_col(df)
    gcol = get_grp_col(df)
    if scol:
        sujs = sorted(df[scol].unique())
        print(f"Subjects ({scol}) (Total N={len(sujs)}): {sujs}")
        if gcol:
            for g in sorted(df[gcol].unique()):
                g_sujs = sorted(df[df[gcol] == g][scol].unique())
                trials_count = len(df[df[gcol] == g])
                print(f"  Group {g} (N={len(g_sujs)}): {g_sujs} -> trials: {trials_count}")
    fcol = 'fase' if 'fase' in df.columns else ('Fase' if 'Fase' in df.columns else None)
    if fcol:
        print("  Fases:", df[fcol].value_counts().to_dict())

print("\n" + "="*60)
print("3. PSD FEATURE MATRICES (prot*_X_psd_norm_*.csv)")
print("="*60)
for fname in sorted(os.listdir(data_dir)):
    if fname.startswith("prot") and "_X_psd_" in fname:
        fpath = os.path.join(data_dir, fname)
        df_tmp = pd.read_csv(fpath)
        print(f"{fname:45s} -> shape: {df_tmp.shape}")

print("\n" + "="*60)
print("4. TOPOPLOT DATASETS (topoplot_*.csv)")
print("="*60)
for fname in sorted(os.listdir(data_dir)):
    if fname.startswith("topoplot_"):
        fpath = os.path.join(data_dir, fname)
        df_topo = pd.read_csv(fpath)
        scol = get_suj_col(df_topo)
        gcol = get_grp_col(df_topo)
        sujs_topo = sorted(df_topo[scol].unique()) if scol else []
        grps_topo = df_topo[gcol].value_counts().to_dict() if gcol else {}
        bands_topo = df_topo['banda'].unique().tolist() if 'banda' in df_topo.columns else []
        print(f"\n--- {fname} ---")
        print(f"Rows: {len(df_topo)} | Channels: {df_topo['canal'].nunique() if 'canal' in df_topo.columns else 'N/A'}")
        if scol:
            print(f"Subjects ({scol}) (N={len(sujs_topo)}): {sujs_topo}")
        else:
            print("No subject column (already aggregated?)")
            print("Columns:", list(df_topo.columns)[:10])
        if grps_topo:
            print(f"Groups breakdown in rows: {grps_topo}")
        print(f"Bands ({len(bands_topo)}): {bands_topo}")
