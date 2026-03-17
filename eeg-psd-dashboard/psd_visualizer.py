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

CHANNELS_MAP = {
    'Cz': (0, 110),
    'C3': (110, 220),
    'C4': (220, 330)
}

def create_psd_subplots(df_meta, df_x, channels_selected, stratify_by, scale='linear', show_bands=False, theme='light'):
    """
    Creates a faceted grid of PSD plots.
    Cols: Channels selected
    Rows: Unique values of the stratification column
    """
    if not channels_selected or df_x.empty:
        return go.Figure()

    # Pre-calculate frequency array mapping
    freq_vector = np.linspace(0, 1000/2, 1025)[:110]
    
    # Stratification logic
    if stratify_by not in df_meta.columns:
        # Fallback if selected stratification doesn't exist
        strat_col = 'All'
        groups = ['All']
        df_meta['All'] = 'All'
    else:
        strat_col = stratify_by
        # Convert to string and clean up floats like '1.0' -> '1'
        df_meta[strat_col] = df_meta[strat_col].astype(str).str.replace(r'\.0$', '', regex=True)
        groups = df_meta[strat_col].unique().tolist()
    
    # Sort groups for consistent rendering
    groups.sort()

    n_rows = len(groups)
    n_cols = len(channels_selected)

    # Calculate global max/min for standardized y-axes (so they don't jump around)
    # But only do this if plotting linear (log scale handles its own range naturally well or gets clamped)
    y_min, y_max = 0, 0
    
    # Create subplot layout
    fig = make_subplots(
        rows=n_rows, cols=n_cols, 
        shared_xaxes=True, shared_yaxes=True,
        subplot_titles=[f"Ch: {ch} | Strata: {g}" for g in groups for ch in channels_selected],
        vertical_spacing=0.08, horizontal_spacing=0.05
    )

    row_idx = 1
    for group in groups:
        # Filter data for this stratum
        mask = df_meta[strat_col] == group
        stratum_x = df_x[mask]
        
        col_idx = 1
        for channel in channels_selected:
            # Extract channel slice
            start_col, end_col = CHANNELS_MAP[channel]
            ch_data = stratum_x.iloc[:, start_col:end_col].values
            
            # Apply log10 scaling if requested
            if scale == 'log10':
                # Small epsilon to avoid log(0)
                ch_data = np.log10(ch_data + 1e-10)
                
            # Plot individual trials
            for trial_idx in range(ch_data.shape[0]):
                trial_data = ch_data[trial_idx, :]
                fig.add_trace(go.Scatter(
                    x=freq_vector, y=trial_data,
                    mode='lines',
                    line=dict(color='rgba(150, 150, 150, 0.15)', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=row_idx, col=col_idx)

            # Calculate Mean and Std Deviation
            mean_psd = np.mean(ch_data, axis=0)
            std_psd = np.std(ch_data, axis=0)
            
            upper_bound = mean_psd + std_psd
            lower_bound = mean_psd - std_psd
            
            # For linear, clamp lower bound at 0 if it goes negative (PSD is strictly positive)
            if scale == 'linear':
                lower_bound = np.maximum(lower_bound, 0)

            # Plot Standard Deviation Area
            fig.add_trace(go.Scatter(
                x=np.concatenate([freq_vector, freq_vector[::-1]]),
                y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill='toself',
                fillcolor='rgba(100, 100, 100, 0.3)' if theme == 'light' else 'rgba(200, 200, 200, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            ), row=row_idx, col=col_idx)

            # Plot Mean Line
            line_color = '#0d6efd' if theme == 'light' else '#4db8ff'
            legend_name = f"Avg PSD ({group})" if n_cols == 1 else f"Avg PSD"
            
            fig.add_trace(go.Scatter(
                x=freq_vector, y=mean_psd,
                mode='lines',
                line=dict(color=line_color, width=2.5),
                name='PSD média dos trials' if row_idx == 1 and col_idx == 1 else '',
                showlegend=True if row_idx == 1 and col_idx == 1 else False
            ), row=row_idx, col=col_idx)

            # 5. Destacar Bandas Mechanism
            if show_bands:
                band_annotations = []
                for b_name, b_info in BANDS.items():
                    f_min, f_max = b_info['range']
                    # Draw a colored vertical rectangle spanning the y-axis
                    fig.add_vrect(
                        x0=f_min, x1=f_max,
                        fillcolor=b_info['color'], opacity=0.5,
                        layer="below", line_width=0,
                        row=row_idx, col=col_idx
                    )
                    
                    # Calculate Band Power (Mean of the Mean Line within the range)
                    mask_freq = (freq_vector >= f_min) & (freq_vector <= f_max)
                    if np.any(mask_freq):
                        band_mean = np.mean(mean_psd[mask_freq])
                        band_std = np.std(mean_psd[mask_freq])
                        band_annotations.append(f"{b_name.capitalize()}: {band_mean:.2f} ± {band_std:.2f}")

                # Attach annotation text
                if band_annotations:
                    annotation_text = "<br>".join(band_annotations)
                    subplot_idx = (row_idx - 1) * n_cols + col_idx
                    xref_str = f"x{subplot_idx if subplot_idx > 1 else ''} domain"
                    yref_str = f"y{subplot_idx if subplot_idx > 1 else ''} domain"
                    
                    # Use a subtle annotation inside the plot
                    fig.add_annotation(
                        x=0.98, y=0.95, xref=xref_str, yref=yref_str,
                        text=annotation_text, showarrow=False, align="right",
                        font=dict(size=10, color="black" if theme == 'light' else "white"),
                        bgcolor="rgba(255,255,255,0.7)" if theme == 'light' else "rgba(0,0,0,0.5)",
                        bordercolor="gray", borderwidth=1, borderpad=4
                    )

            col_idx += 1
        row_idx += 1

    # Formatting and Theming
    bg_color = '#2d2d2d' if theme == 'dark' else 'white'
    font_color = '#e0e0e0' if theme == 'dark' else '#212529'
    grid_color = '#444' if theme == 'dark' else '#e9ecef'

    fig.update_layout(
        height=300 * n_rows, # Dynamically scale height based on number of strata
        margin=dict(l=50, r=20, t=60, b=50),
        template='plotly_dark' if theme == 'dark' else 'plotly_white',
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=font_color),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(title_text="Frequência (Hz)" if row_idx > 1 else "", showgrid=True, gridcolor=grid_color)
    fig.update_yaxes(title_text=f"Potência {'(Log10)' if scale == 'log10' else ''}", showgrid=True, gridcolor=grid_color)

    return fig
