import pandas as pd
import numpy as np
import os
import io
import base64
import matplotlib
import matplotlib.pyplot as plt
import mne
import scipy.stats as stats
from data_loader import get_condition_n

# Force matplotlib to use non-interactive backend for server compatibility
matplotlib.use('Agg')

# Strict dictionary to normalize user raw strings to exactly what MNE standard_1020 expects for the 32 channels.
# MNE's standard_1020 expects e.g. "Fp1", "Cz", "Pz". 
CH_MAPPING = {
    'FP1': 'Fp1', 'FP2': 'Fp2', 'FZ': 'Fz', 'F3': 'F3', 'F4': 'F4', 'F7': 'F7', 'F8': 'F8',
    'CZ': 'Cz', 'C3': 'C3', 'C4': 'C4', 'T7': 'T7', 'T8': 'T8', 'P7': 'P7', 'P8': 'P8',
    'PZ': 'Pz', 'P3': 'P3', 'P4': 'P4', 'O1': 'O1', 'O2': 'O2', 'FCZ': 'FCz', 'FC1': 'FC1',
    'FC2': 'FC2', 'FC3': 'FC3', 'FC4': 'FC4', 'OZ': 'Oz', 'C1': 'C1', 'C2': 'C2',
    'CP1': 'CP1', 'CP2': 'CP2', 'CP3': 'CP3', 'CP4': 'CP4', 'CPZ': 'CPz'
}

BANDS_ORDER = ['total', 'delta', 'theta', 'alfa', 'beta', 'gamma']

def get_topoplot_path(protocol, fase, is_normalized, is_baseline):
    """Resolve correct data file path for topoplot based on UI selections"""
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    
    if is_baseline:
        # User defined filename format for baseline (no phase suffix)
        file_path = os.path.join(data_dir, "topoplot_baseline_olhosAbertos_protC.csv")
        return file_path

    if protocol in ['A', 'B']:
        file_path = os.path.join(data_dir, f"topoplot_prot{protocol}_{fase}_norm.csv")
    elif protocol == 'C':
        # Prot C can be normalizado or cru
        norm_suffix = "_norm" if is_normalized else ""
        file_path = os.path.join(data_dir, f"topoplot_protC_{fase}{norm_suffix}.csv")
    else:
        file_path = None
        
    return file_path

def generate_topoplot_grid_base64(protocol, fase, group, scale_db, is_normalized=True, is_baseline=False):
    """
    Reads the relevant file, filters, constructs a 1x6 matplotlib figure using mne, 
    and returns a base64 png string.
    """
    file_path = get_topoplot_path(protocol, fase, is_normalized, is_baseline)
    fname = os.path.basename(file_path) if file_path else 'Caminho nulo'
    
    print(f"[DEBUG ENGINE] Carregando topoplot: {fname} | Protocol: {protocol} | Group: {group}")
    
    if not file_path or not os.path.exists(file_path):
         return None, f"Arquivo de dados não encontrado: {fname}"
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return None, f"Erro ao ler CSV: {str(e)}"
        
    # Coluna alvo e tratamentos base
    target_col = 'psd_db_mean' if scale_db else 'psd_mean'
    if target_col not in df.columns:
         return None, f"A coluna alvo {target_col} não existe no arquivo."
         
    # Filtrar grupo caso protocolo A ou B
    if protocol in ['A', 'B'] and not is_baseline:
        if 'grupo' in df.columns:
            df = df[df['grupo'] == group]
            if df.empty:
                return None, f"Grupo {group} não localizado no dataset {fname} (fase {fase})."
        else:
            return None, "Dados não possuem a coluna 'grupo' para filtro."

    # Padronizar nomes das bandas, pois podem existir grafias diferentes (ex: alfa vs alpha)
    if 'banda' in df.columns:
         df['banda'] = df['banda'].str.lower().replace({'alpha': 'alfa'})
    
    # Criar 1x6 matplotlib figure
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    montage = mne.channels.make_standard_montage('standard_1020')
    
    # Renderizar para cada banda
    for i, band in enumerate(BANDS_ORDER):
        ax = axes[i]
        
        # Obter o valor da banda especifica; 'total' muitas vezes pode não vir como lowercase, fallback robusto
        df_band = df[df['banda'] == band].copy()
        
        if df_band.empty:
            ax.set_title(band.capitalize())
            ax.axis('off')
            ax.text(0.5, 0.5, "Sem Dados", ha='center', va='center')
            continue
            
        # Limpar os canais normalizando para o MNE Standard 1020
        df_band['mne_canal'] = df_band['canal'].str.upper().map(CH_MAPPING)

        # Tratar eventuais canais órfãos que não batem com o map, skip the bad ones
        df_band = df_band.dropna(subset=['mne_canal'])
        
        ch_names = df_band['mne_canal'].tolist()
        vals = df_band[target_col].values
        
        if len(ch_names) == 0:
            ax.set_title(band.capitalize())
            ax.axis('off')
            ax.text(0.5, 0.5, "Sem Canais Válidos", ha='center', va='center')
            continue

        try:
            info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')
            info.set_montage(montage)
            
            # Ajustar limites vmin vmax 
            vmin, vmax = df_band[target_col].min(), df_band[target_col].max()
            
            im, cm = mne.viz.plot_topomap(
                vals, info, axes=ax, show=False, 
                cmap='RdBu_r', 
                vlim=(vmin, vmax),
                extrapolate='head', 
                sphere=0.095, # Standard radius to keep it inside the head circle
                contours=4
            )
            ax.set_title(band.capitalize())
            
            # Anexar colorbar apenas para dar uma guia visual rapida
            fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
            
        except Exception as mape:
            ax.set_title(f"{band.capitalize()} (Error)")
            ax.axis('off')
            print(f"Erro Plot Mne banda {band}: {str(mape)}")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=120)
    plt.close(fig)
    
    return base64.b64encode(buf.getvalue()).decode('utf-8'), None

def generate_topoplot_comparison_base64(p1, p2, scale_db):
    """
    Computes t-tests between two topoplot conditions and plots the difference map.
    p1, p2 are dicts with {protocol, fase, group, is_normalized, is_baseline}
    """
    # 1. Load data for both panels
    f1 = get_topoplot_path(p1['protocol'], p1['fase'], p1['is_normalized'], p1['is_baseline'])
    f2 = get_topoplot_path(p2['protocol'], p2['fase'], p2['is_normalized'], p2['is_baseline'])
    
    if not os.path.exists(f1) or not os.path.exists(f2):
        return None, "Um dos arquivos de topoplot não foi localizado."
        
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    
    # Target columns
    m_col = 'psd_db_mean' if scale_db else 'psd_mean'
    s_col = 'psd_db_std' if scale_db else 'psd_std'
    
    # Sample sizes
    n1 = get_condition_n(p1['protocol'], p1['group'])
    n2 = get_condition_n(p2['protocol'], p2['group'])
    
    # Standard filtering for A/B
    if p1['protocol'] in ['A', 'B'] and not p1['is_baseline']:
        df1 = df1[df1['grupo'] == p1['group']]
    if p2['protocol'] in ['A', 'B'] and not p2['is_baseline']:
        df2 = df2[df2['grupo'] == p2['group']]

    if 'banda' in df1.columns: df1['banda'] = df1['banda'].str.lower().replace({'alpha': 'alfa'})
    if 'banda' in df2.columns: df2['banda'] = df2['banda'].str.lower().replace({'alpha': 'alfa'})

    fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))
    montage = mne.channels.make_standard_montage('standard_1020')
    
    stats_data = []
    
    for i, band in enumerate(BANDS_ORDER):
        ax = axes[i]
        d1 = df1[df1['banda'] == band].copy()
        d2 = df2[df2['banda'] == band].copy()
        
        band_summary = {'band': band.capitalize(), 'channels': []}
        
        if d1.empty or d2.empty:
            ax.set_title(band.capitalize())
            ax.axis('off')
            stats_data.append(band_summary)
            continue

        # Merge on channel to ensure alignment
        d1['mne_canal'] = d1['canal'].str.upper().map(CH_MAPPING)
        d2['mne_canal'] = d2['canal'].str.upper().map(CH_MAPPING)
        merged = pd.merge(d1, d2, on='mne_canal', suffixes=('_1', '_2'))
        merged = merged.dropna(subset=['mne_canal'])
        
        if merged.empty:
            ax.axis('off')
            stats_data.append(band_summary)
            continue

        ch_names = merged['mne_canal'].tolist()
        mean1, std1 = merged[m_col + '_1'].values, merged[s_col + '_1'].values
        mean2, std2 = merged[m_col + '_2'].values, merged[s_col + '_2'].values
        
        # Calculate T-test
        t_stat, p_vals = stats.ttest_ind_from_stats(
            mean1=mean1, std1=std1, nobs1=n1,
            mean2=mean2, std2=std2, nobs2=n2,
            equal_var=False
        )
        
        # Determine difference and mask
        diff_vals = mean1 - mean2
        mask = p_vals < 0.05
        
        # Collect significant channels
        curr_channels = []
        for idx in range(len(ch_names)):
            if mask[idx]:
                curr_channels.append({
                    'ch': ch_names[idx],
                    'p': float(p_vals[idx])
                })
        band_summary['channels'] = curr_channels
        stats_data.append(band_summary)
        
        try:
            info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')
            info.set_montage(montage)
            
            # Use symmetric color limits for difference maps
            vabs = np.max(np.abs(diff_vals)) if len(diff_vals) > 0 else 1
            im, _ = mne.viz.plot_topomap(
                diff_vals, info, axes=ax, show=False, 
                cmap='RdBu_r', vlim=(-vabs, vabs),
                extrapolate='head', sphere=0.095,
                mask=mask, 
                mask_params=dict(marker='x', markerfacecolor='black', markersize=8, markeredgecolor='black', markeredgewidth=2)
            )
            ax.set_title(f"{band.capitalize()}\n(Δ Mean)")
            fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7, pad=0.1)
        except Exception as e:
            ax.axis('off')
            print(f"Error in comparison plot ({band}): {e}")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8'), stats_data, f"T-Test (Panel 1 vs 2) | n1={n1}, n2={n2} | p < 0.05 highlighted with X."
