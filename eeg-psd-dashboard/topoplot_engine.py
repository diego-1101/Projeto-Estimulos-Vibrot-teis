import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
import base64
import io
import os
from data_loader import get_condition_n

# Global Cache for Topoplot bounds
_TOPO_BOUNDS_CACHE = {}

matplotlib.use('Agg')

CH_MAPPING = {
    'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'F3': 'F3', 'F4': 'F4', 'F7': 'F7', 'F8': 'F8',
    'CZ': 'Cz', 'C3': 'C3', 'C4': 'C4', 'T7': 'T7', 'T8': 'T8', 'P7': 'P7', 'P8': 'P8',
    'PZ': 'Pz', 'P3': 'P3', 'P4': 'P4', 'O1': 'O1', 'O2': 'O2', 'FCZ': 'FCz', 'FC1': 'FC1',
    'FC2': 'FC2', 'FC3': 'FC3', 'FC4': 'FC4', 'OZ': 'Oz', 'C1': 'C1', 'C2': 'C2',
    'CP1': 'CP1', 'CP2': 'CP2', 'CP3': 'CP3', 'CP4': 'CP4', 'CPZ': 'CPz'
}

BANDS_ORDER = ['total', 'delta', 'theta', 'alfa', 'beta', 'gamma']

TRIAL_COUNTS = {
    'A': {'CV': 486, 'SV': 567, 'estimulacao': 1053, 'execucao': 1053},
    'B': {'CF': 135, 'SF': 135, 'estimulacao': 270, 'execucao': 270},
    'C': {'estimulacao': 108, 'execucao': 144}
}

def get_topoplot_path(protocol, fase, is_normalized, is_baseline):
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    if is_baseline: return os.path.join(data_dir, "topoplot_baseline_olhosAbertos_protC.csv")
    if protocol in ['A', 'B']: return os.path.join(data_dir, f"topoplot_prot{protocol}_{fase}_norm.csv")
    elif protocol == 'C':
        norm_suffix = "_norm" if is_normalized else ""
        return os.path.join(data_dir, f"topoplot_protC_{fase}{norm_suffix}.csv")
    return None

def get_topoplot_bounds(mode, protocol=None, group=None, target_col='psd_db_mean'):
    """Calculates min and max values for topoplot scaling based on the selected mode."""
    import glob
    global _TOPO_BOUNDS_CACHE
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    
    files = []
    if mode == 'global':
        files = glob.glob(os.path.join(data_dir, "topoplot_*.csv"))
    elif mode == 'protocol' and protocol:
        clean_prot = 'C' if 'baseline' in protocol.lower() else protocol
        files = glob.glob(os.path.join(data_dir, f"topoplot_prot{clean_prot}_*.csv"))
        if clean_prot == 'C':
            files.append(os.path.join(data_dir, "topoplot_baseline_olhosAbertos_protC.csv"))
    
    cache_key = f"{mode}_{protocol}_{group}_{target_col}"
    if cache_key in _TOPO_BOUNDS_CACHE: return _TOPO_BOUNDS_CACHE[cache_key]

    all_vals = []
    import glob
    for f in files:
        if not os.path.exists(f): continue
        try:
            df = pd.read_csv(f)
            if group and 'grupo' in df.columns: df = df[df['grupo'] == group]
            if target_col in df.columns: all_vals.extend(df[target_col].dropna().values)
        except: pass
    
    if not all_vals: return -10, 10
    vmin, vmax = np.min(all_vals), np.max(all_vals)
    vrange = vmax - vmin if vmax > vmin else 1.0
    res = (vmin - 0.05*vrange, vmax + 0.05*vrange)
    _TOPO_BOUNDS_CACHE[cache_key] = res
    return res

def combine_topoplot_data(dfs, protocols, groups, phases, target_col, std_col=None):
    if not dfs: return pd.DataFrame()
    for df in dfs:
        if 'banda' in df.columns: df['banda'] = df['banda'].str.lower().replace({'alpha': 'alfa'})
    
    phase_results = []
    for df, prot, group, phase in zip(dfs, protocols, groups, phases):
        if group == 'Ambos' and prot in ['A', 'B']:
            g1, g2 = ('CV', 'SV') if prot == 'A' else ('CF', 'SF')
            n1, n2 = TRIAL_COUNTS[prot][g1], TRIAL_COUNTS[prot][g2]
            d1 = df[df['grupo'] == g1].copy().set_index(['canal', 'banda'])
            d2 = df[df['grupo'] == g2].copy().set_index(['canal', 'banda'])
            if not d1.empty and not d2.empty:
                combined_vals = (d1[target_col] * n1 + d2[target_col] * n2) / (n1 + n2)
                res_df = combined_vals.reset_index()
                if std_col and std_col in d1.columns:
                    combined_var = (n1*(d1[std_col]**2 + d1[target_col]**2) + n2*(d2[std_col]**2 + d2[target_col]**2)) / (n1+n2) - combined_vals**2
                    res_df[std_col] = np.sqrt(combined_var.clip(lower=0)).values
                phase_results.append((res_df, n1 + n2))
            elif not d1.empty: phase_results.append((d1.reset_index(), n1))
            elif not d2.empty: phase_results.append((d2.reset_index(), n2))
        else:
            if 'grupo' in df.columns and group and group != 'Ambos': df = df[df['grupo'] == group].copy()
            elif 'grupo' in df.columns and prot == 'C' and not group: df = df[df['grupo'].isin(['all', 'ALL'])].copy()
            n = TRIAL_COUNTS['C'].get(phase, 144) if prot == 'C' else (TRIAL_COUNTS[prot]['estimulacao'] if group == 'Ambos' else TRIAL_COUNTS[prot].get(group, 270))
            if not df.empty: phase_results.append((df, n))

    if not phase_results: return pd.DataFrame()
    if len(phase_results) == 1: return phase_results[0][0]

    total_n = sum(n for _, n in phase_results)
    base_df = phase_results[0][0].copy().set_index(['canal', 'banda'])
    weighted_sum = base_df[target_col] * phase_results[0][1]
    for i in range(1, len(phase_results)):
        df_next = phase_results[i][0].set_index(['canal', 'banda'])
        weighted_sum += df_next[target_col] * phase_results[i][1]
    
    final_mean = (weighted_sum / total_n).reset_index()
    if std_col and std_col in phase_results[0][0].columns:
        sum_ni_vi_mi2 = 0
        for df, n in phase_results:
            df_idx = df.set_index(['canal', 'banda'])
            sum_ni_vi_mi2 += n * (df_idx[std_col]**2 + df_idx[target_col]**2)
        final_var = (sum_ni_vi_mi2 / total_n) - (final_mean.set_index(['canal', 'banda'])[target_col]**2)
        final_mean[std_col] = np.sqrt(final_var.clip(lower=0)).values
    return final_mean


def calculate_anova_p_from_stats(means, stds, ns):
    import scipy.stats as stats
    k = len(means)
    N_total = sum(ns)
    if N_total <= k: return 1.0
    grand_mean = sum(m * n for m, n in zip(means, ns)) / N_total
    ss_between = sum(n * (m - grand_mean)**2 for m, n in zip(means, ns))
    ss_within = sum((n - 1) * s**2 for s, n in zip(stds, ns))
    df_between, df_within = k - 1, N_total - k
    ms_between, ms_within = ss_between / df_between, ss_within / df_within
    if ms_within <= 0: return 1.0
    return 1 - stats.f.cdf(ms_between / ms_within, df_between, df_within)

def generate_topoplot_grid_base64(protocol, fase, group, scale_db, is_normalized=True, is_baseline=False, vmin=None, vmax=None, band_limits=None):
    fases_to_load = ['estimulacao', 'execucao'] if fase == 'Ambos' else [fase]
    dfs_to_combine = []
    for f in fases_to_load:
        path = get_topoplot_path(protocol, f, is_normalized, is_baseline)
        if path and os.path.exists(path): dfs_to_combine.append(pd.read_csv(path))
    if not dfs_to_combine: return None, "Dados não localizados."
    
    t_col = 'psd_db_mean' if scale_db else 'psd_mean'
    s_col = 'psd_db_std' if scale_db else 'psd_std'
    df = combine_topoplot_data(dfs_to_combine, [protocol]*len(dfs_to_combine), [group]*len(dfs_to_combine), fases_to_load, t_col, s_col)
    
    fig, axes = plt.subplots(1, 6, figsize=(15, 4), facecolor='none')
    montage = mne.channels.make_standard_montage('standard_1020')
    for i, band in enumerate(BANDS_ORDER):
        df_b = df[df['banda'] == band]
        ch_names = [CH_MAPPING.get(c.upper(), c) for c in df_b['canal']]
        data = df_b[t_col].values
        info = mne.create_info(ch_names, sfreq=100, ch_types='eeg')
        info.set_montage(montage)
        
        b_vmin, b_vmax = vmin, vmax
        if band_limits and band in band_limits: b_vmin, b_vmax = band_limits[band]
        
        im, _ = mne.viz.plot_topomap(data, info, axes=axes[i], show=False, contours=0, cmap='RdBu_r', vlim=(b_vmin, b_vmax))
        axes[i].set_title(band.capitalize(), fontsize=10)
        fig.colorbar(im, ax=axes[i], orientation='horizontal', fraction=0.046, pad=0.15)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8'), None

def generate_topoplot_comparison_base64(p1, p2, scale_db, label1="Panel 1", label2="Panel 2"):
    import scipy.stats as stats
    def get_dfs(p):
        flist = ['estimulacao', 'execucao'] if p['fase'] == 'Ambos' else [p['fase']]
        res = []
        for f in flist:
            path = get_topoplot_path(p['protocol'], f, p['is_normalized'], p['is_baseline'])
            if path and os.path.exists(path): res.append(pd.read_csv(path))
        return res, flist

    dfs1, f1_list = get_dfs(p1)
    dfs2, f2_list = get_dfs(p2)
    if not dfs1 or not dfs2: return None, "Dados não localizados."
    
    m_col, s_col = ('psd_db_mean', 'psd_db_std') if scale_db else ('psd_mean', 'psd_std')
    from data_loader import get_condition_n
    def get_n(p):
        if p['fase'] == 'Ambos' and p['group'] == 'Ambos': return get_condition_n(p['protocol'], 'Ambos') * 2
        elif p['fase'] == 'Ambos' or p['group'] == 'Ambos': return get_condition_n(p['protocol'], p['group']) * 2 if p['fase'] == 'Ambos' else get_condition_n(p['protocol'], 'Ambos')
        return get_condition_n(p['protocol'], p['group'])
    n1, n2 = get_n(p1), get_n(p2)
    
    df1 = combine_topoplot_data(dfs1, [p1['protocol']]*len(dfs1), [p1['group']]*len(dfs1), f1_list, m_col, s_col)
    df2 = combine_topoplot_data(dfs2, [p2['protocol']]*len(dfs2), [p2['group']]*len(dfs2), f2_list, m_col, s_col)
    
    all_ps, comp_results = [], []
    for band in BANDS_ORDER:
        d1b, d2b = df1[df1['banda'] == band].set_index('canal'), df2[df2['banda'] == band].set_index('canal')
        chs = d1b.index.intersection(d2b.index)
        m1, s1, m2, s2 = d1b.loc[chs, m_col].values, d1b.loc[chs, s_col].values, d2b.loc[chs, m_col].values, d2b.loc[chs, s_col].values
        _, p_vals = stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False)
        all_ps.extend(p_vals)
        comp_results.append((band, chs, m1 - m2, p_vals))
    
    import statsmodels.stats.multitest as mt
    is_sig_global = []
    
    for band, chs, diff, pvals in comp_results:
        # T-test pode retornar NaNs se os desvios padrão forem zero, vamos tratar isso
        valid_pvals = np.nan_to_num(pvals, nan=1.0) 
        if len(valid_pvals) > 0:
            reject, _, _, _ = mt.multipletests(valid_pvals, alpha=0.05, method='fdr_bh')
            is_sig_global.extend(reject.tolist())
        else:
            is_sig_global.extend([False] * len(valid_pvals))
            
    is_sig_global = np.array(is_sig_global, dtype=bool)
    
    fig, axes = plt.subplots(1, 6, figsize=(15, 4), facecolor='none')
    montage = mne.channels.make_standard_montage('standard_1020')
    sig_info = {}
    sig_ptr = 0
    for i, (band, chs, diff, pvals) in enumerate(comp_results):
        ch_names = [CH_MAPPING.get(c.upper(), c) for c in chs]
        info = mne.create_info(ch_names, sfreq=100, ch_types='eeg')
        info.set_montage(montage)
        
        mask = is_sig_global[sig_ptr : sig_ptr + len(chs)]
        sig_ptr += len(chs)
        
        import matplotlib.colors as mcolors
        white_cmap = mcolors.ListedColormap(['white'])
        v_zeros = np.zeros(len(diff))
        
        im, _ = mne.viz.plot_topomap(v_zeros, info, axes=axes[i], show=False, contours=0, cmap=white_cmap, vlim=(0, 1), mask=np.array(mask), mask_params=dict(marker='x', markeredgecolor='black', markersize=6))
        axes[i].set_title(band.capitalize(), fontsize=10)
        # fig.colorbar(im, ax=axes[i], orientation='horizontal', fraction=0.046, pad=0.15)
        
        band_sig = []
        for ch, p, s in zip(chs, pvals, mask):
             if s: band_sig.append({'canal': ch, 'p': p})
        if band_sig: sig_info[band] = band_sig

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8'), sig_info

def generate_anova_map_base64(panels_data, scale_db):
    import scipy.stats as stats
    m_col, s_col = ('psd_db_mean', 'psd_db_std') if scale_db else ('psd_mean', 'psd_std')
    processed_dfs, ns = [], []
    for p in panels_data:
        flist = ['estimulacao', 'execucao'] if p['fase'] == 'Ambos' else [p['fase']]
        dfs = [pd.read_csv(get_topoplot_path(p['protocol'], f, p['is_normalized'], p['is_baseline'])) for f in flist if os.path.exists(get_topoplot_path(p['protocol'], f, p['is_normalized'], p['is_baseline']))]
        df_c = combine_topoplot_data(dfs, [p['protocol']]*len(dfs), [p['group']]*len(dfs), flist, m_col, s_col)
        processed_dfs.append(df_c)
        from data_loader import get_condition_n
        n = get_condition_n(p['protocol'], p['group'])
        if p['fase'] == 'Ambos': n *= 2
        ns.append(n)
        
    all_ps, anova_results = [], []
    for band in BANDS_ORDER:
        b_dfs = [df[df['banda'] == band].set_index('canal') for df in processed_dfs]
        chs = b_dfs[0].index
        for bdf in b_dfs[1:]: chs = chs.intersection(bdf.index)
        
        pvals = []
        for ch in chs:
            ms = [bdf.loc[ch, m_col] for bdf in b_dfs]
            ss = [bdf.loc[ch, s_col] for bdf in b_dfs]
            pvals.append(calculate_anova_p_from_stats(ms, ss, ns))
        all_ps.extend(pvals)
        anova_results.append((band, chs, pvals))

    import statsmodels.stats.multitest as mt
    
    is_sig_global = []
    # Aplicando FDR (False Discovery Rate) por banda para correção de múltiplas comparações
    for band, chs, pvals in anova_results:
        # Se tiver p-valores, ajustamos. Se não, tudo Falso.
        if len(pvals) > 0:
            reject, pvals_corrected, _, _ = mt.multipletests(pvals, alpha=0.05, method='fdr_bh')
            is_sig_global.extend(reject.tolist())
        else:
            is_sig_global.extend([False] * len(pvals))
            
    is_sig_global = np.array(is_sig_global, dtype=bool)
    
    fig, axes = plt.subplots(1, 6, figsize=(15, 4), facecolor='none')
    montage = mne.channels.make_standard_montage('standard_1020')
    sig_ptr = 0
    anova_details = {} # Para o post-hoc ou análise detalhada depois
    for i, (band, chs, pvals) in enumerate(anova_results):
        ch_names = [CH_MAPPING.get(c.upper(), c) for c in chs]
        info = mne.create_info(ch_names, sfreq=100, ch_types='eeg')
        info.set_montage(montage)
        mask = is_sig_global[sig_ptr : sig_ptr + len(chs)]
        sig_ptr += len(chs)
        
        # Armazenar canais significativos para possível uso do Front-End
        sig_channels = [ch for ch, is_sig in zip(chs, mask) if is_sig]
        if sig_channels:
            anova_details[band] = sig_channels
            
        import matplotlib.colors as mcolors
        white_cmap = mcolors.ListedColormap(['white'])
        v_zeros = np.zeros(len(pvals))
        
        im, _ = mne.viz.plot_topomap(v_zeros, info, axes=axes[i], show=False, contours=0, cmap=white_cmap, vlim=(0, 1), mask=np.array(mask), mask_params=dict(marker='x', markeredgecolor='black', markersize=6))
        axes[i].set_title(band.capitalize(), fontsize=10)
        # fig.colorbar(im, ax=axes[i], orientation='horizontal', fraction=0.046, pad=0.15)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8'), anova_details

def get_channel_reference_base64():
    """Generates a reference map of the 32 electrodes used."""
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='none')
    montage = mne.channels.make_standard_montage('standard_1020')
    ch_names = list(CH_MAPPING.values())
    info = mne.create_info(ch_names, sfreq=100, ch_types='eeg')
    info.set_montage(montage)
    mne.viz.plot_sensors(info, show_names=True, axes=ax, show=False)
    ax.set_title("32-Channel Layout (10-20)", fontsize=10)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')
