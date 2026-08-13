import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Standard Bands setup
BANDS = {
    'delta': {'range': (0.5, 4), 'color': 'rgba(255, 99, 132, 0.2)'},
    'theta': {'range': (4, 8), 'color': 'rgba(54, 162, 235, 0.2)'},
    'alpha': {'range': (8, 13), 'color': 'rgba(255, 206, 86, 0.2)'},
    'beta': {'range': (13, 30), 'color': 'rgba(75, 192, 192, 0.2)'},
    'gamma': {'range': (30, 55), 'color': 'rgba(153, 102, 255, 0.2)'}
}

# Dynamic channel handling used below based on column names.

def create_psd_subplots(df_meta, df_x, channels_selected, stratify_by, scale='linear', show_bands=False, overlay_strata=False, theme='light'):
    """
    Creates a faceted grid of PSD plots.
    Cols: Channels selected
    Rows: Unique values of the stratification column (or 1 if overlaying strata)
    """
    if not channels_selected or df_x.empty:
        return go.Figure()

    # Pre-calculate frequency array mapping
    freq_vector = np.linspace(0, 1000/2, 1025)[:110]
    
    # Stratification logic
    if stratify_by not in df_meta.columns:
        strat_col = 'All'
        groups = ['All']
        df_meta['All'] = 'All'
    else:
        strat_col = stratify_by
        df_meta[strat_col] = df_meta[strat_col].astype(str).str.replace(r'\.0$', '', regex=True)
        groups = df_meta[strat_col].unique().tolist()
    
    groups.sort()

    palette = ['#0d6efd', '#dc3545', '#198754', '#fd7e14', '#6f42c1', '#20c997', '#e83e8c', '#0dcaf0']

    n_rows = 1 if overlay_strata else len(groups)
    n_cols = len(channels_selected)

    global_min, global_max = float('inf'), float('-inf')
    
    subplot_titles = [f"Ch: {ch}" for ch in channels_selected] if overlay_strata else [f"Ch: {ch} | Strata: {g}" for g in groups for ch in channels_selected]

    fig = make_subplots(
        rows=n_rows, cols=n_cols, 
        shared_xaxes=True, shared_yaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.15, horizontal_spacing=0.05
    )

    bg_color = '#2d2d2d' if theme == 'dark' else 'white'
    font_color = '#e0e0e0' if theme == 'dark' else '#212529'

    if overlay_strata:
        row_idx = 1
        for col_idx, channel in enumerate(channels_selected, start=1):
            
            for g_idx, group in enumerate(groups):
                mask = df_meta[strat_col] == group
                stratum_x = df_x[mask]
                if stratum_x.empty: continue
                
                # Fetch dynamically the columns corresponding to this channel
                channel_cols = [c for c in stratum_x.columns if c.split('_')[0] == channel]
                if not channel_cols: continue
                ch_data = stratum_x[channel_cols].values
                if scale == 'log10':
                    ch_data = np.log10(ch_data + 1e-10)
                
                mean_psd = np.mean(ch_data, axis=0)
                
                global_min = min(global_min, np.min(mean_psd))
                global_max = max(global_max, np.max(mean_psd))
                
                line_color = palette[g_idx % len(palette)]
                
                fig.add_trace(go.Scatter(
                    x=freq_vector, y=mean_psd,
                    mode='lines',
                    line=dict(color=line_color, width=2.5),
                    name=f'{group}',
                    showlegend=True if col_idx == 1 else False,
                    legendgroup=group
                ), row=row_idx, col=col_idx)

            if show_bands:
                for b_name, b_info in BANDS.items():
                    f_min, f_max = b_info['range']
                    fig.add_vrect(
                        x0=f_min, x1=f_max,
                        fillcolor=b_info['color'], opacity=0.8,
                        layer="below", line_width=0,
                        row=row_idx, col=col_idx
                    )
    else:
        row_idx = 1
        for group in groups:
            mask = df_meta[strat_col] == group
            stratum_x = df_x[mask]
            
            col_idx = 1
            for channel in channels_selected:
                # Fetch dynamically the columns corresponding to this channel
                channel_cols = [c for c in stratum_x.columns if c.split('_')[0] == channel]
                if not channel_cols:
                     col_idx += 1
                     continue
                ch_data = stratum_x[channel_cols].values
                
                if scale == 'log10':
                    ch_data = np.log10(ch_data + 1e-10)
                    
                for trial_idx in range(ch_data.shape[0]):
                    trial_data = ch_data[trial_idx, :]
                    fig.add_trace(go.Scatter(
                        x=freq_vector, y=trial_data,
                        mode='lines',
                        line=dict(color='rgba(150, 150, 150, 0.15)', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ), row=row_idx, col=col_idx)

                mean_psd = np.mean(ch_data, axis=0)
                std_psd = np.std(ch_data, axis=0)
                
                upper_bound = mean_psd + std_psd
                lower_bound = mean_psd - std_psd
                
                if scale == 'linear':
                    lower_bound = np.maximum(lower_bound, 0)

                global_min = min(global_min, np.min(lower_bound))
                global_max = max(global_max, np.max(upper_bound))

                fig.add_trace(go.Scatter(
                    x=np.concatenate([freq_vector, freq_vector[::-1]]),
                    y=np.concatenate([upper_bound, lower_bound[::-1]]),
                    fill='toself',
                    fillcolor='rgba(100, 100, 100, 0.3)' if theme == 'light' else 'rgba(200, 200, 200, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                ), row=row_idx, col=col_idx)

                line_color = '#0d6efd' if theme == 'light' else '#4db8ff'
                
                fig.add_trace(go.Scatter(
                    x=freq_vector, y=mean_psd,
                    mode='lines',
                    line=dict(color=line_color, width=2.5),
                    name='PSD média dos trials',
                    showlegend=True if row_idx == 1 and col_idx == 1 else False,
                    legendgroup='mean_line'
                ), row=row_idx, col=col_idx)

                if show_bands:
                    band_annotations = []
                    for b_name, b_info in BANDS.items():
                        f_min, f_max = b_info['range']
                        fig.add_vrect(
                            x0=f_min, x1=f_max,
                            fillcolor=b_info['color'], opacity=0.8,
                            layer="below", line_width=0,
                            row=row_idx, col=col_idx
                        )
                        
                        mask_freq = (freq_vector >= f_min) & (freq_vector <= f_max)
                        if np.any(mask_freq):
                            band_mean = np.mean(mean_psd[mask_freq])
                            band_std = np.std(mean_psd[mask_freq])
                            band_annotations.append(f"{b_name.capitalize()}: {band_mean:.2f} ± {band_std:.2f}")

                    if band_annotations:
                        annotation_text = "<br>".join(band_annotations)
                        subplot_idx = (row_idx - 1) * n_cols + col_idx
                        xref_str = f"x{subplot_idx if subplot_idx > 1 else ''} domain"
                        yref_str = f"y{subplot_idx if subplot_idx > 1 else ''} domain"
                        
                        fig.add_annotation(
                            x=0.98, y=0.95, xref=xref_str, yref=yref_str,
                            text=annotation_text, showarrow=False, align="right",
                            font=dict(size=10, color="black" if theme == 'light' else "white"),
                            bgcolor="rgba(255,255,255,0.7)" if theme == 'light' else "rgba(0,0,0,0.5)",
                            bordercolor="gray", borderwidth=1, borderpad=4
                        )

                col_idx += 1
            row_idx += 1

    # Add band legends (Fake traces)
    if show_bands:
        for b_name, b_info in BANDS.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=12, color=b_info['color'].replace('0.2', '0.8'), symbol='square'),
                name=f"{b_name.capitalize()}",
                showlegend=True,
                legendgroup="bands"
            ))

    # Formatting and Theming
    grid_color = '#444' if theme == 'dark' else '#e9ecef'

    fig.update_layout(
        height=350 * n_rows, # Scaled height with padding 
        margin=dict(l=60, r=20, t=60, b=50),
        template='plotly_dark' if theme == 'dark' else 'plotly_white',
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=font_color),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1,
            bgcolor='rgba(255,255,255,0)' # Transparent legend
        )
    )

    # Force shared Y-axis exactly across all subplots
    if global_max > global_min:
        padding = (global_max - global_min) * 0.05
        fig.update_yaxes(range=[global_min - padding, global_max + padding])

    # Ensure X axes look correct (only bottom row gets labels by default, or 1 row)
    if overlay_strata or n_rows == 1:
        fig.update_xaxes(title_text="Frequência (Hz)", showgrid=True, gridcolor=grid_color)
    else:
        fig.update_xaxes(showgrid=True, gridcolor=grid_color)
        for c in range(1, n_cols + 1):
             fig.update_xaxes(title_text="Frequência (Hz)", row=n_rows, col=c)
             
    fig.update_yaxes(title_text=f"Potência {'(Log10)' if scale == 'log10' else ''}", showgrid=True, gridcolor=grid_color, col=1)

    return fig
