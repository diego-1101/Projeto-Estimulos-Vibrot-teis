import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import dash
from dash import dcc, html, Input, Output, State, ALL, MATCH
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Import our modules
from data_loader import load_data, build_X, build_Y
from analysis_engine import compute_embeddings
from anova_engine import compute_anova_and_plot
from psd_visualizer import create_psd_subplots
from topoplot_engine import (
    generate_topoplot_grid_base64, 
    generate_topoplot_comparison_base64, 
    get_channel_reference_base64, 
    get_topoplot_bounds,
    generate_inverted_topoplot_grid_base64
)
import os

# Load Quick Guide Content
base_path = os.path.dirname(__file__)

from performance_engine import (
    run_normality_test, run_anova_typ3, plot_interactive_dot_sig, 
    plot_interactive_interaction, plot_interactive_hybrid, plot_interactive_significance_heatmap
)

try:
    with open(os.path.join(base_path, 'QUICK_GUIDE.md'), 'r', encoding='utf-8') as f:
        quick_guide_content = f.read()
except FileNotFoundError:
    quick_guide_content = "Quick Guide not found. Please ensure QUICK_GUIDE.md is in the eeg-psd-dashboard folder."

try:
    with open(os.path.join(base_path, 'TOPO_GUIDE.md'), 'r', encoding='utf-8') as f:
        topo_guide_content = f.read()
except FileNotFoundError:
    topo_guide_content = "Topoplot Guide not found. Please ensure TOPO_GUIDE.md is in the eeg-psd-dashboard folder."

# --- Globals & Setup ---
app = dash.Dash(
    __name__, 
    external_stylesheets=['https://bootswatch.com/5/flatly/bootstrap.min.css'],
    suppress_callback_exceptions=True
)
app.title = "EEG PSD Dashboard"

data_cache = {}
current_theme = 'light'

# Define standard variables available for Y Checklist
Y_VARIABLES = [
    {'label': 'Desempenho', 'value': 'Desempenho'},
    {'label': 'Acurácia', 'value': 'Acuracia'},
    {'label': 'Similaridade', 'value': 'Similaridade'},
    {'label': 'Especificidade', 'value': 'Especificidade'},
    {'label': 'Proporção Espacial X', 'value': 'Proporção espacial x'},
    {'label': 'Proporção Espacial Y', 'value': 'Proporção espacial y'},
]

ALL_CHANNELS = ['FP1', 'FP2', 'FZ', 'F3', 'F4', 'F7', 'F8', 'CZ', 'C3', 'C4', 'T7', 'T8', 'P7', 'P8', 'PZ', 'P3', 'P4', 'O1', 'O2', 'FCZ', 'FC1', 'FC2', 'FC3', 'OZ', 'C2', 'CP1', 'CP3', 'CP4', 'C1', 'FC4', 'CPZ', 'CP2']
CHANNELS_C = [ch for ch in ALL_CHANNELS if ch not in ['CP2', 'CP4']]

# --- Helper Functions ---

def build_supervision_labels(meta, protocol, color_by_mode):
    """
    Constructs the supervision labels categorically based on protocol limits.
    If the requested mode is invalid for the given protocol, it falls back to
    a safe default and indicates a warning.
    """
    labels = []
    warning = False
    mode = color_by_mode

    # Verify Protocol A options (group, complexity, overlap, and combinations)
    if protocol == 'A':
        valid_modes = ['group', 'complexity', 'overlap', 'group_comp', 'group_overlap', 'comp_overlap', 'all']
        if mode not in valid_modes:
            mode = 'group'
            warning = True

    # Verify Protocol B options (group, complexity, group+complexity)
    elif protocol == 'B':
        valid_modes = ['group', 'complexity', 'group_comp']
        if mode not in valid_modes:
            mode = 'group'
            warning = True

    # Verify Protocol C options (complexity only; group=='ALL' exists but shouldn't be separated)
    elif protocol == 'C':
        valid_modes = ['complexity']
        if mode not in valid_modes:
            mode = 'complexity'
            warning = True

    # Assemble series
    try:
        group = meta['grupo'].astype(str)
        comp = meta.get('Complexidade', pd.Series('Unk', index=meta.index)).astype(str).str.replace(r'\.0$', '', regex=True)
        over = meta.get('Overlap', pd.Series('Unk', index=meta.index)).astype(str).str.replace(r'\.0$', '', regex=True)
        
        if mode == 'group':
            color_s = group
            symbol_s = group
            cluster_s = group
        elif mode == 'complexity':
            color_s = 'C' + comp
            symbol_s = color_s
            cluster_s = color_s
        elif mode == 'overlap':
            color_s = 'O' + over
            symbol_s = color_s
            cluster_s = color_s
        elif mode == 'group_comp':
            color_s = 'C' + comp
            symbol_s = group
            cluster_s = group + '_C' + comp
        elif mode == 'group_overlap':
            color_s = 'O' + over
            symbol_s = group
            cluster_s = group + '_O' + over
        elif mode == 'comp_overlap':
            color_s = 'O' + over
            symbol_s = 'C' + comp
            cluster_s = 'C' + comp + '_O' + over
        elif mode == 'all':
            color_s = 'C' + comp + '_O' + over
            symbol_s = group
            cluster_s = group + '_C' + comp + '_O' + over
        
        color_labels = color_s.tolist()
        symbol_labels = symbol_s.tolist()
        cluster_labels = cluster_s.tolist()
    except KeyError as e:
        # Fallback if a column is miraculously missing
        print(f"KeyError: {e}. Falling back to default Group coloring.")
        color_labels = meta['grupo'].astype(str).tolist()
        symbol_labels = color_labels
        cluster_labels = color_labels
        warning = True
        
    return color_labels, symbol_labels, cluster_labels, warning

def run_projected_anova(coords_df, color_labels, target_label="Projected Axis 1"):
    """
    Calculates one-way ANOVA on the first projected dimension (the plotted points).
    """
    temp_df = pd.DataFrame({
        'val': coords_df.iloc[:, 0].values,
        'label': color_labels
    })
    from anova_engine import compute_anova_and_plot
    fig, stats = compute_anova_and_plot(temp_df, 'val', 'label')
    # Update axis label in title for clarity
    fig.update_layout(title=f"ANOVA: {target_label}")
    return fig, stats

def create_analysis_controls(panel_id):
    """Create a set of analysis controls for a panel."""
    return html.Div([
        html.H5(f"Analysis {panel_id}", className="text-primary mb-3"),
        
        html.Label("Analysis Method", className="control-label"),
        dcc.Dropdown(
            id={'type': 'method-dropdown', 'index': panel_id},
            options=[
                {'label': 'PCA', 'value': 'PCA'},
                {'label': 'LDA', 'value': 'LDA'},
                {'label': 'CDA', 'value': 'CDA'},
                {'label': 'PLS', 'value': 'PLS'}
            ],
            value='PCA',
            className="mb-3 dash-dropdown"
        ),

        html.Label("Fase", className="control-label"),
        dcc.Dropdown(id={'type': 'fase-dropdown', 'index': panel_id}, className="mb-3 dash-dropdown"),
        
        html.Label("X (PSD Features)", className="control-label"),
        dcc.Dropdown(
            id={'type': 'x-mode-dropdown', 'index': panel_id},
            options=[
                {'label': 'PSD Completo Normalizado (Baseline)', 'value': 'psd_full_norm'},
                {'label': 'PSD Potência de 2 em 2 Hz Normalizado', 'value': 'psd_2em2_norm'}
            ],
            value='psd_full_norm',
            className="mb-3 dash-dropdown"
        ),

        html.Label("Canais (Features do X)", className="control-label"),
        dcc.Dropdown(
            id={'type': 'channels-dropdown', 'index': panel_id},
            options=[{'label': 'Select All', 'value': 'ALL'}] + [{'label': ch, 'value': ch} for ch in ALL_CHANNELS],
            value=['CZ', 'C3', 'C4'],
            multi=True,
            className="mb-3 dash-dropdown"
        ),

        html.Label("Y (Behavioral Features)", className="control-label"),
        dcc.Checklist(
            id={'type': 'y-checklist', 'index': panel_id},
            options=Y_VARIABLES,
            value=['Desempenho'],
            className="mb-3 list-style-none",
            inline=False,
            labelStyle={'display': 'block', 'marginBottom': '5px'}
        ),
        
        html.Label("Data Domain (Axes mapping)", className="control-label"),
        dcc.Dropdown(
            id={'type': 'domain-dropdown', 'index': panel_id},
            options=[
                {'label': 'X Only (PSD)', 'value': 'x'},
                {'label': 'Y Only (Behavior)', 'value': 'y'},
                {'label': 'Both (Mix Axes)', 'value': 'both'}
            ],
            value='x',
            className="mb-3 dash-dropdown"
        ),
        
        html.Div(id={'type': 'mixed-axes-container', 'index': panel_id}, className="mb-3", style={'display': 'none'}, children=[
            html.Label("Axis 1:", className="control-label"),
            dcc.Dropdown(id={'type': 'axis-select', 'index': panel_id, 'axis': 1}, className="mb-2 dash-dropdown"),
            html.Label("Axis 2:", className="control-label"),
            dcc.Dropdown(id={'type': 'axis-select', 'index': panel_id, 'axis': 2}, className="mb-2 dash-dropdown"),
            html.Div(id={'type': 'axis3-container', 'index': panel_id}, style={'display': 'none'}, children=[
                 html.Label("Axis 3:", className="control-label"),
                 dcc.Dropdown(id={'type': 'axis-select', 'index': panel_id, 'axis': 3}, className="mb-2 dash-dropdown"),
            ])
        ]),
        
        html.Label("Color By (Visual)", className="control-label"),
        dcc.Dropdown(
            id={'type': 'color-dropdown', 'index': panel_id},
            options=[
                {'label': 'Group (CV/SV/CF/SF)', 'value': 'group'},
                {'label': 'Complexity', 'value': 'complexity'},
                {'label': 'Overlap (Prot A)', 'value': 'overlap'},
                {'label': 'Group + Complexity', 'value': 'group_comp'},
                {'label': 'Group + Overlap', 'value': 'group_overlap'},
                {'label': 'Complexity + Overlap', 'value': 'comp_overlap'},
                {'label': 'Group + Comp + Overlap', 'value': 'all'}
            ],
            value='group',
            className="mb-3 dash-dropdown"
        ),

        # Supervision is only for CDA/LDA
        html.Div(id={'type': 'supervision-container', 'index': panel_id}, style={'display': 'none'}, children=[
            html.Label("Supervision (CDA/LDA Math)", className="control-label"),
            dcc.Dropdown(
                id={'type': 'supervision-dropdown', 'index': panel_id},
                options=[
                    {'label': 'Group (CV/SV)', 'value': 'grupo'},
                    {'label': 'Complexity', 'value': 'complexity'},
                    {'label': 'Overlap (Prot A)', 'value': 'overlap'}
                ],
                value='grupo',
                className="mb-3 dash-dropdown"
            ),
        ]),
        
        html.Hr(),
    ], className="comparison-panel mb-3" if panel_id == 2 else "mb-3")

def run_single_analysis(protocol, groups_selected, method, x_mode, y_cols, domain, axes, n_dims, theme='light', color_by='group', supervision_by='complexity', fase='estimulacao', selected_channels=None):
    """Run analysis and return figure, stats."""
    try:
        # Load caching
        if protocol not in data_cache:
            df, meta = load_data(protocol=protocol)
            data_cache[protocol] = (df, meta)
        else:
            df, meta = data_cache[protocol]

        # 1. Filter by Selected Groups
        if groups_selected:
            mask = df['grupo'].isin(groups_selected)
            df = df[mask].copy()
            meta = meta[mask].copy()

        if df.empty:
            raise ValueError("No data available for the selected groups.")

        # Validations before continuing
        if not y_cols:
            raise ValueError("Please select at least one Y variable from the checklist.")

        # Build feature matrices
        X = build_X(df, x_mode, fase=fase, selected_channels=selected_channels)
        
        # Ensure metadata is aligned with the loaded features (X might have fewer rows)
        df = df.loc[X.index].copy()
        meta = meta.loc[X.index].copy()
        
        Y = build_Y(df, y_cols)
        
        # Build supervision labels
        # 1. Labels for Visual Coloring
        color_labels, symbol_labels, cluster_labels, _ = build_supervision_labels(meta, protocol, color_by)
        
        # 2. Labels for Mathematical Supervision (CDA/LDA)
        _, _, math_labels_colors, had_warning = build_supervision_labels(meta, protocol, supervision_by)
        math_labels = math_labels_colors

        # Compute Embeddings using math_labels for supervision
        X_scores, Y_scores, stats = compute_embeddings(X, Y, math_labels, method, max(3, n_dims))
        
        # Build Plot Coordinates
        coords_df = pd.DataFrame(index=X.index)
        axis_names = []
        
        if domain == 'x':
             if X_scores is not None and not X_scores.empty:
                 cols_to_use = min(n_dims, X_scores.shape[1])
                 for i in range(cols_to_use):
                     axis_name = f"{method} {i+1} (X)"
                     coords_df[axis_name] = X_scores.iloc[:, i]
                     axis_names.append(axis_name)
                 while len(coords_df.columns) < n_dims:
                     axis_name = f"Empty {len(coords_df.columns)+1}"
                     coords_df[axis_name] = 0
                     axis_names.append(axis_name)
             else:
                 raise ValueError(f"No valid X embedding formed for method {method}.")
                 
        elif domain == 'y':
             if Y_scores is not None and not Y_scores.empty:
                 cols_to_use = min(n_dims, Y_scores.shape[1])
                 for i in range(cols_to_use):
                     axis_name = f"{method} {i+1} (Y)"
                     coords_df[axis_name] = Y_scores.iloc[:, i]
                     axis_names.append(axis_name)
                 while len(coords_df.columns) < n_dims:
                     axis_name = f"Empty {len(coords_df.columns)+1}"
                     coords_df[axis_name] = 0
                     axis_names.append(axis_name)
             else:
                 raise ValueError(f"No valid Y embedding formed for method {method}.")
                 
        elif domain == 'both':
             for d in range(n_dims):
                 # default map if no axis list supplied
                 axis_sel = axes[d] if axes and len(axes) > d and axes[d] else None
                 val = 0
                 if axis_sel:
                      parts = axis_sel.split('_') # e.g. "C1_X"
                      comp_idx = int(parts[0].replace('C', '')) - 1
                      source = parts[1]
                      
                      if source == 'X' and X_scores is not None and comp_idx < X_scores.shape[1]:
                          val = X_scores.iloc[:, comp_idx]
                      elif source == 'Y' and Y_scores is not None and comp_idx < Y_scores.shape[1]:
                          val = Y_scores.iloc[:, comp_idx]
                          
                 axis_name = axis_sel.replace('_', ' of ') if axis_sel else f"Dim{d+1}"
                 coords_df[axis_name] = val
                 axis_names.append(axis_name)
                 
        # Combine coords with meta for plotting
        plot_df = pd.concat([coords_df, meta], axis=1)
        plot_df['color_label'] = color_labels
        plot_df['symbol_label'] = symbol_labels
        plot_df['cluster_label'] = cluster_labels
        
        # --- Projected ANOVA ---
        # Calculate ANOVA on the first plotted dimension vs cluster labels
        # we must drop NaNs for the stat test
        stat_df = plot_df[[axis_names[0], 'cluster_label']].dropna()
        try:
            if not stat_df.empty and len(stat_df['cluster_label'].unique()) > 1:
                from anova_engine import compute_anova_and_plot
                anova_fig, anova_stats = compute_anova_and_plot(stat_df, axis_names[0], 'cluster_label')
                anova_fig.update_layout(title=f"ANOVA: {axis_names[0]}")
                if theme == 'dark':
                    anova_fig.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                anova_res_text = f"F: {anova_stats.get('F', 0):.2f} | p-value: {anova_stats.get('p_value', 1):.4f}"
            else:
                anova_fig, anova_res_text = go.Figure(), "ANOVA: Insufficient groups/data"
        except Exception as ae:
            anova_fig, anova_res_text = go.Figure(), f"ANOVA Error: {str(ae)}"

        # --- Centroid Distances ---
        centroid_res = []
        try:
            if not plot_df.empty and len(plot_df['cluster_label'].unique()) > 1:
                # Group by cluster_label to get the centroids in the current projection
                coord_cols = [c for c in axis_names if pd.api.types.is_numeric_dtype(plot_df[c])]
                centroids = plot_df.groupby('cluster_label')[coord_cols].mean()
                
                pairs = []
                c_names = centroids.index.tolist()
                for i in range(len(c_names)):
                    for j in range(i+1, len(c_names)):
                        p1 = centroids.loc[c_names[i]].values
                        p2 = centroids.loc[c_names[j]].values
                        dist = np.linalg.norm(p1 - p2)
                        pairs.append((c_names[i], c_names[j], dist))
                
                # Sort by distance 
                pairs.sort(key=lambda x: x[2], reverse=True)
                
                for g1, g2, d in pairs:
                    centroid_res.append(
                        html.Div(f"{g1} ↔ {g2} : {d:.3f}", 
                                 className="badge bg-secondary me-2 mb-2 p-2", 
                                 style={'fontSize': '0.9em'})
                    )
            
            if not centroid_res:
                centroid_res = html.P("Not enough groups to compare centroids.", className="text-muted mb-0", style={'fontSize': '0.9em'})
        except Exception as ce:
            centroid_res = html.Div(f"Centroid Error: {str(ce)}", className="text-danger")

        title_suffix = f"{color_by.capitalize()}" + (" (Warning)" if had_warning else "")

        # Base Hover Configs
        hover_cols = ['ID', 'grupo']
        # Dynamically add info what's available
        if 'Complexidade' in meta.columns: hover_cols.append('Complexidade')
        if 'Overlap' in meta.columns: hover_cols.append('Overlap')
        for yc in y_cols:
             target_raw = f"raw_{yc}"
             if target_raw in meta.columns:
                 hover_cols.append(target_raw)

        # Plot definitions
        color_map = None
        if color_by == 'group' and protocol in ['A', 'B']:
            color_map = {'CV': '#2ecc71', 'SV': '#e74c3c', 'CF': '#2ecc71', 'SF': '#e74c3c'}
            
        symbol_map = {'CV': 'circle', 'SV': 'square', 'CF': 'circle', 'SF': 'square'}
            
        params = {
             'color': 'color_label',
             'symbol': 'symbol_label',
             'symbol_map': symbol_map,
             'hover_data': hover_cols,
             'title': f"{method} - {domain.upper()} - {title_suffix}"
        }
        if color_map:
             params['color_discrete_map'] = color_map

        if n_dims == 2:
            fig = px.scatter(plot_df, x=axis_names[0], y=axis_names[1], **params)
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
        else:
            fig = px.scatter_3d(plot_df, x=axis_names[0], y=axis_names[1], z=axis_names[2], **params)
            fig.update_traces(marker=dict(size=6))
        
        # Theme
        if theme == 'dark':
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#2d2d2d',
                plot_bgcolor='#2d2d2d',
                font=dict(color='#e0e0e0')
            )
        else:
            fig.update_layout(template='plotly_white')
        
        # Format Stats Panel
        stats_content = []
        
        if 'X_explained_variance' in stats:
             var_txt = ", ".join([f"C{i+1}: {v*100:.1f}%" for i, v in enumerate(stats['X_explained_variance'])])
             stats_content.append(html.P([html.Strong("X Var: "), var_txt]))
        if 'Y_explained_variance' in stats:
             var_txt = ", ".join([f"C{i+1}: {v*100:.1f}%" for i, v in enumerate(stats['Y_explained_variance'])])
             stats_content.append(html.P([html.Strong("Y Var: "), var_txt]))
        if 'canonical_correlations' in stats:
             corr_txt = ", ".join([f"r{i+1}: {c:.3f}" for i, c in enumerate(stats['canonical_correlations'])])
             stats_content.append(html.P([html.Strong("Canonical Corrs: "), corr_txt]))
             
        if 'CDA_X' in stats:
             cdx = stats['CDA_X']
             chisqs = ", ".join([f"{c:.1f}" for c in cdx['chisq']])
             stats_content.append(html.P([html.Strong("CDA X Ext. Dims: "), str(cdx['d'])]))
             stats_content.append(html.P([html.Strong("CDA X ChiSqs: "), chisqs]))
             if 'explained_variance_ratio' in cdx:
                 var_txt = ", ".join([f"CD{i+1}: {v*100:.1f}%" for i, v in enumerate(cdx['explained_variance_ratio'])])
                 stats_content.append(html.P([html.Strong("CDA X Var Explicada: "), var_txt]))
             if 'eigenval' in cdx:
                 eig_txt = ", ".join([f"CD{i+1}: {v:.4f}" for i, v in enumerate(cdx['eigenval'])])
                 stats_content.append(html.P([html.Strong("CDA X Eigenvalues: "), eig_txt]))

        if 'CDA_Y' in stats:
             cdy = stats['CDA_Y']
             chisqs_y = ", ".join([f"{c:.1f}" for c in cdy['chisq']])
             stats_content.append(html.P([html.Strong("CDA Y Ext. Dims: "), str(cdy['d'])]))
             stats_content.append(html.P([html.Strong("CDA Y ChiSqs: "), chisqs_y]))
             if 'explained_variance_ratio' in cdy:
                 var_txt_y = ", ".join([f"CD{i+1}: {v*100:.1f}%" for i, v in enumerate(cdy['explained_variance_ratio'])])
                 stats_content.append(html.P([html.Strong("CDA Y Var Explicada: "), var_txt_y]))
             if 'eigenval' in cdy:
                 eig_txt_y = ", ".join([f"CD{i+1}: {v:.4f}" for i, v in enumerate(cdy['eigenval'])])
                 stats_content.append(html.P([html.Strong("CDA Y Eigenvalues: "), eig_txt_y]))

        # We must return 5 elements
        return fig, html.Div(stats_content or "No stats evaluated"), anova_fig, anova_res_text, html.Div(centroid_res, className="d-flex flex-wrap")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Error: {str(e)}")
        return error_fig, html.Div(f"Error: {str(e)}"), go.Figure(), "", ""


def get_performance_layout():
    return html.Div([
        html.Div([
            html.H2("EEG PSD Dashboard", className="text-primary mb-4"),
            html.H5("v2 Performance Evaluation", className="text-muted mb-4"),
            html.Hr(),
            
            html.Label("Protocol", className="control-label"),
            dcc.Dropdown(
                id='perf-protocol-dropdown',
                options=[
                    {'label': 'Protocol A', 'value': 'A'},
                    {'label': 'Protocol B', 'value': 'B'},
                    {'label': 'Protocol C', 'value': 'C'}
                ],
                value='A',
                className="mb-3 dash-dropdown"
            ),
            
            
            
            html.Hr(),
            
            dcc.Checklist(
                id='perf-normality-check',
                options=[{'label': ' Testar Normalidade', 'value': 'yes'}],
                value=[],
                className="mb-2"
            ),
            html.Div(id='perf-normality-controls', style={'display': 'none'}, children=[
                html.Label("Variável de Teste", className="control-label"),
                dcc.Dropdown(id='perf-normality-var', options=Y_VARIABLES, value='Desempenho', className="mb-2 dash-dropdown"),
                html.Div(id='perf-normality-group-container', children=[
                    html.Label("Grupo", className="control-label"),
                    dcc.Dropdown(id='perf-normality-group', options=[{'label': 'CV', 'value': 'CV'}, {'label': 'SV', 'value': 'SV'}, {'label': 'Ambos', 'value': 'Ambos'}], value='Ambos', className="mb-2 dash-dropdown")
                ])
            ]),
            
            html.Hr(),
            
            dcc.Checklist(
                id='perf-anova-check',
                options=[{'label': ' Executar ANOVA Typ=3 & Post-Hoc', 'value': 'yes'}],
                value=[],
                className="mb-2"
            ),
            html.Div(id='perf-anova-controls', style={'display': 'none'}, children=[
                html.Label("Variável Dependente (Y)", className="control-label"),
                dcc.Dropdown(id='perf-anova-target', options=Y_VARIABLES, value='Desempenho', className="mb-2 dash-dropdown"),
                html.Label("Variáveis Independentes (X)", className="control-label"),
                dcc.Checklist(id='perf-anova-independents', options=[
                    {'label': 'Complexidade', 'value': 'Complexidade'},
                    {'label': 'Grupo', 'value': 'grupo'},
                    {'label': 'Overlap', 'value': 'Overlap'}
                ], value=['Complexidade'], className="mb-2 list-style-none", labelStyle={'display': 'block', 'marginBottom': '5px'}),
                dcc.Checklist(id='perf-heatmap-check', options=[{'label': ' Exibir Heatmap de Significância', 'value': 'yes'}], value=[], className="mt-2 mb-2")
            ]),
            
            html.Hr(),
            
            html.Div(id='perf-alpha-container', style={'display': 'none'}, children=[
                html.Label("Nível de Significância (Alpha)", className="control-label"),
                dcc.Input(id='perf-alpha-input', type='number', value=0.05, min=0.001, max=0.1, step=0.01, className="form-control mb-3")
            ]),

            dcc.Checklist(
                id='perf-interaction-check',
                options=[{'label': ' Plotar Interaction Plot', 'value': 'yes'}],
                value=[],
                className="mb-2"
            ),
            dcc.Checklist(
                id='perf-hybrid-check',
                options=[{'label': ' Plotar Gráfico Híbrido (Dispersão + Interação)', 'value': 'yes'}],
                value=[],
                className="mb-2"
            ),
            html.Div(id='perf-interaction-controls', style={'display': 'none'}, children=[
                html.Label("Eixo X", className="control-label"),
                dcc.Dropdown(id='perf-interaction-x', options=[], className="mb-2 dash-dropdown"),
                html.Label("Eixo Y", className="control-label"),
                dcc.Dropdown(id='perf-interaction-y', options=Y_VARIABLES, value='Desempenho', className="mb-2 dash-dropdown"),
                html.Label("Linha", className="control-label"),
                dcc.Dropdown(id='perf-interaction-line', options=[], className="mb-2 dash-dropdown"),
                html.Label("Facet", className="control-label"),
                dcc.Dropdown(id='perf-interaction-facet', options=[{'label': 'None', 'value': 'None'}], value='None', className="mb-2 dash-dropdown"),
                html.Label("Escala Y (ex: 0.2, 1.0)", className="control-label"),
                dcc.Input(id='perf-interaction-ylim', type='text', value='(0.2, 1.0)', className="form-control mb-2")
            ]),
            
            html.Hr(),
            html.Button("Executar Análise", id='perf-run-btn', className="btn btn-primary w-100 mt-2"),
            html.Div(id='perf-controls-error', className="text-danger mt-2")
        ], className="sidebar"),
        
        html.Div([
            html.Div(id='perf-normality-result-card', style={'display': 'none'}, className="card p-3 mb-3 shadow-sm", children=[
                html.H4("Teste de Normalidade", className="text-info"),
                html.Div(id='perf-normality-output')
            ]),
            
            html.Div(id='perf-anova-result-card', style={'display': 'none'}, className="card p-3 mb-3 shadow-sm", children=[
                html.H4("ANOVA e Post-Hoc", className="text-warning"),
                html.Div(id='perf-anova-table-output', style={'overflowX': 'auto'}),
                dcc.Graph(id='perf-anova-dotplot', style={'height': '600px', 'marginTop': '15px'}),
                html.Div(id='perf-anova-heatmap-container', style={'display': 'none'}, children=[
                    html.H5("Heatmap de Significância (Tukey HSD)", className="text-primary mt-3"),
                    dcc.Graph(id='perf-anova-heatmap', style={'height': '500px', 'marginTop': '10px'})
                ]),
                html.Details([
                    html.Summary("Tabela de Resultados Post-Hoc (Significativos)", style={'cursor': 'pointer', 'fontWeight': 'bold', 'color': '#0d6efd', 'marginTop': '15px'}),
                    html.Div(id='perf-anova-posthoc-output', className="mt-2 bg-light p-2 rounded", style={'overflowX': 'auto'})
                ])
            ]),
            
            html.Div(id='perf-interaction-result-card', style={'display': 'none'}, className="card p-3 mb-3 shadow-sm", children=[
                html.H4("Interaction Plot", className="text-success"),
                dcc.Graph(id='perf-interaction-plot', style={'height': '500px'})
            ]),

            html.Div(id='perf-hybrid-result-card', style={'display': 'none'}, className="card p-3 mb-3 shadow-sm", children=[
                html.H4("Gráfico Híbrido (Dispersão + Interação)", className="text-primary"),
                dcc.Graph(id='perf-hybrid-plot', style={'height': '550px'})
            ])

            
        ], className="main-content")
    ])

# --- Layout ---

def get_analysis_layout():
    return html.Div([
        html.Div([
            html.H2("EEG PSD Dashboard", className="text-primary mb-4"),
            html.H5("v2 Multivariate Analysis", className="text-muted mb-4"),
        html.Hr(),
        
        html.Label("Protocol", className="control-label"),
        dcc.Dropdown(
            id='protocol-dropdown',
            options=[
                {'label': 'Protocol A', 'value': 'A'},
                {'label': 'Protocol B', 'value': 'B'},
                {'label': 'Protocol C', 'value': 'C'}
            ],
            value='A',
            className="mb-3 dash-dropdown"
        ),
        
        html.Div(id='group-filter-container', children=[
            html.Label("Groups", className="control-label"),
            dcc.Checklist(
                id='group-checklist',
                options=[{'label': 'CV', 'value': 'CV'}, {'label': 'SV', 'value': 'SV'}],
                value=['CV', 'SV'],
                className="mb-3 list-style-none",
                inline=False,
                labelStyle={'display': 'block', 'marginRight': '10px'}
            ),
        ]),
        
        html.Label("Dimensions", className="control-label"),
        dcc.RadioItems(
            id='global-dimensions-radio',
            options=[
                {'label': ' 2D', 'value': 2},
                {'label': ' 3D', 'value': 3}
            ],
            value=2,
            className="mb-3"
        ),
        
        html.Hr(),
        
        html.Label("Comparison Mode", className="control-label"),
        dcc.Checklist(
            id='comparison-toggle',
            options=[{'label': ' Enable Comparison', 'value': 'yes'}],
            value=[],
            className="mb-3"
        ),
        
        html.Hr(),
        html.Div(id='controls-1', children=create_analysis_controls(1)),
        html.Div(id='controls-2-container'),
        
        html.Details([
            html.Summary("📐 Display Mathematical Model", style={'cursor': 'pointer', 'fontWeight': 'bold', 'color': '#0d6efd', 'marginBottom': '15px'}),
            html.Div(id='math-model-container', className="card p-3 mb-3", style={'backgroundColor': '#f8f9fa', 'overflowX': 'auto', 'fontSize': '0.9em'})
        ], className="mb-3"),
        
        html.Button("Run Analysis", id='run-btn', className="btn btn-primary w-100 mt-2"),
        html.Hr(),
        html.Div(id='info-panel', className="card p-3 mt-3")
        
    ], className="sidebar"),
    
    html.Div([
        html.Div(id='single-view', children=[
            html.Div([
                html.Div([
                    html.Button("ℹ️ Quick Guide", id={'type': 'guide-btn', 'index': 'analysis-single'}, className='btn btn-sm btn-outline-info')
                ], style={'position': 'absolute', 'top': '10px', 'left': '15px', 'zIndex': 500}),
                dcc.Graph(id='plot-1', style={'height': '600px'})
            ], className="card mb-3", style={'position': 'relative'}),
            html.Div([html.H4("Statistics"), html.Div(id='stats-1')], className="card mb-3"),
            html.Div([
                html.H4("ANOVA Test"),
                html.Div(id='anova-stats-1', className="mb-2"),
                dcc.Graph(id='anova-plot-1', style={'height': '500px'})
            ], className="card mb-3"),
            html.Div([
                html.H4("Centroid Distances"),
                html.Div(id='centroid-distance-1', className="mb-2")
            ], className="card")
        ]),
        
        html.Div(id='comparison-view', style={'display': 'none'}, children=[
            html.Div(className="comparison-container", children=[
                html.Div([
                    html.Div([
                        html.Div([
                            html.Button("📖 Quick Guide", id={'type': 'guide-btn', 'index': 'analysis-left'}, className='btn btn-sm btn-outline-info')
                        ], style={'position': 'absolute', 'top': '10px', 'left': '15px', 'zIndex': 500}),
                        dcc.Graph(id='plot-left', style={'height': '500px'})
                    ], className="card mb-3", style={'position': 'relative'}),
                    html.Div([html.H5("Stats 1"), html.Div(id='stats-left')], className="card mt-3 mb-3"),
                    html.Div([
                        html.H5("ANOVA 1"),
                        html.Div(id='anova-stats-left', className="mb-2"),
                        dcc.Graph(id='anova-plot-left', style={'height': '400px'})
                    ], className="card mb-3"),
                    html.Div([
                        html.H5("Centroid Distances 1"), 
                        html.Div(id='centroid-distance-left', className="mb-2")
                    ], className="card")
                ]),
                html.Div([
                    html.Div([
                        html.Div([
                            html.Button("📖 Quick Guide", id={'type': 'guide-btn', 'index': 'analysis-right'}, className='btn btn-sm btn-outline-info')
                        ], style={'position': 'absolute', 'top': '10px', 'left': '15px', 'zIndex': 500}),
                        dcc.Graph(id='plot-right', style={'height': '500px'})
                    ], className="card mb-3", style={'position': 'relative'}),
                    html.Div([html.H5("Stats 2"), html.Div(id='stats-right')], className="card mt-3 mb-3"),
                    html.Div([
                        html.H5("ANOVA 2"),
                        html.Div(id='anova-stats-right', className="mb-2"),
                        dcc.Graph(id='anova-plot-right', style={'height': '400px'})
                    ], className="card mb-3"),
                    html.Div([
                        html.H5("Centroid Distances 2"), 
                        html.Div(id='centroid-distance-right', className="mb-2")
                    ], className="card")
                ])
            ])
        ])
    ], className="main-content", id='main-content')
    ])

def get_psd_layout():
    return html.Div([
        html.Div([
            html.H2("EEG PSD Dashboard", className="text-primary mb-4"),
            html.H5("v2 PSD Visualization", className="text-muted mb-4"),
            html.Hr(),
            
            html.Label("Protocol", className="control-label"),
            dcc.Dropdown(
                id='psd-protocol-dropdown',
                options=[
                    {'label': 'Protocol A', 'value': 'A'},
                    {'label': 'Protocol B', 'value': 'B'},
                    {'label': 'Protocol C', 'value': 'C'}
                ],
                value='A',
                className="mb-3 dash-dropdown"
            ),
            html.Label("Fase", className="control-label"),
            dcc.Dropdown(id='psd-fase-dropdown', className="mb-3 dash-dropdown"),
            
            html.Label("Channels", className="control-label"),
            dcc.Dropdown(
                id='psd-channels-checklist',
                options=[{'label': 'Select All', 'value': 'ALL'}] + [{'label': ch, 'value': ch} for ch in ALL_CHANNELS],
                value=['CZ', 'C3', 'C4'],
                multi=True,
                className="mb-3 dash-dropdown"
            ),
            
            html.Label("Scale", className="control-label"),
            dcc.RadioItems(
                id='psd-scale-radio',
                options=[
                    {'label': ' Linear', 'value': 'linear'},
                    {'label': ' Log10', 'value': 'log10'}
                ],
                value='log10',
                className="mb-3"
            ),
            
            html.Label("Frequency Bands", className="control-label"),
            dcc.Checklist(
                id='psd-bands-toggle',
                options=[{'label': ' Highlight Bands', 'value': 'yes'}],
                value=['yes'],
                className="mb-3"
            ),
            
            html.Hr(),
            html.Label("Stratification Label", className="control-label"),
            dcc.Dropdown(
                id='psd-stratify-dropdown',
                options=[
                    {'label': 'Group (CV/SV)', 'value': 'grupo'},
                    {'label': 'Complexity', 'value': 'complexity'},
                    {'label': 'Overlap', 'value': 'overlap'}
                ],
                value='grupo',
                className="mb-3 dash-dropdown"
            ),
            
            dcc.Checklist(
                id='psd-overlay-toggle',
                options=[{'label': ' Overlap Condition (Mean Only)', 'value': 'yes'}],
                value=[],
                className="mb-4 text-muted",
                style={'fontSize': '0.9em'}
            ),
            
            html.Button("Run PSD", id='run-psd-btn', className="btn btn-primary w-100"),
            html.Hr(),
            html.Div(id='psd-info-panel', className="card p-3 mt-3")
        ], className="sidebar"),
        
        html.Div([
            html.Div([
                html.Div([
                    html.H4("PSD Channels Over Frequency", className="mb-3")
                ]),
                # Using dcc.Loading to show a spinner during the initial plot calculation
                dcc.Loading(
                    id="loading-psd",
                    type="default",
                    children=dcc.Graph(id='psd-main-plot', style={'minHeight': '600px'})
                )
            ], className="card p-3 mb-3")
        ], className="main-content")
    ])

def get_topoplot_layout():
    return html.Div([
        html.Div([
            html.H2("EEG PSD Dashboard", className="text-primary mb-4"),
            html.H5("Topoplot Spatial Projections", className="text-muted mb-4"),
            html.Hr(),
            
            html.Details([
                html.Summary("🔧 Options - Panel 1", style={'cursor': 'pointer', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                html.Div([
                    html.Label("Protocol", className="control-label"),
                    dcc.Dropdown(
                        id={'type': 'topo-prot-dropdown', 'index': 1},
                        options=[
                            {'label': 'Protocol A', 'value': 'A'},
                            {'label': 'Protocol B', 'value': 'B'},
                            {'label': 'Protocol C', 'value': 'C'},
                            {'label': 'Baseline (Protocol C)', 'value': 'baseline_C'}
                        ],
                        value='A',
                        className="mb-3 dash-dropdown"
                    ),
                    
                    html.Div(id={'type': 'topo-fase-container', 'index': 1}, children=[
                        html.Label("Fase", className="control-label"),
                        dcc.Dropdown(id={'type': 'topo-fase-dropdown', 'index': 1}, className="mb-3 dash-dropdown"),
                    ]),
                    
                    html.Div(id={'type': 'topo-group-container', 'index': 1}, children=[
                        html.Label("Group", className="control-label"),
                        dcc.Dropdown(id={'type': 'topo-group-dropdown', 'index': 1}, className="mb-3 dash-dropdown")
                    ]),
                    
                    html.Div(id={'type': 'topo-norm-container', 'index': 1}, children=[
                        dcc.Checklist(
                            id={'type': 'topo-norm-check', 'index': 1},
                            options=[{'label': ' Normalizado', 'value': 'yes'}],
                            value=['yes'],
                            className="mb-3"
                        )
                    ], style={'display': 'none'}),
                    
                    html.Label("Scale", className="control-label"),
                    dcc.RadioItems(
                        id={'type': 'topo-scale-radio', 'index': 1},
                        options=[
                            {'label': ' Linear (psd_mean)', 'value': 'linear'},
                            {'label': ' dB (psd_db_mean)', 'value': 'db'}
                        ],
                        value='db',
                        className="mb-3"
                    )
                ], className="p-2 border rounded")
            ], open=True, className="mb-3"),


            html.Label("Número de Plots", className="control-label"),
            dcc.RadioItems(
                id='topo-plot-count-radio',
                options=[
                    {'label': ' 1 Plot', 'value': 1},
                    {'label': ' 2 Plots', 'value': 2},
                    {'label': ' 3 Plots', 'value': 3}
                ],
                value=1,
                className="mb-3"
            ),
            
            html.Div(id='topo-controls-2-wrapper', style={'display':'none'}, children=[
                html.Details([
                    html.Summary("🔧 Options - Panel 2", style={'cursor': 'pointer', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                    html.Div([
                        html.Label("Protocol", className="control-label"),
                        dcc.Dropdown(
                            id={'type': 'topo-prot-dropdown', 'index': 2},
                            options=[
                                {'label': 'Protocol A', 'value': 'A'},
                                {'label': 'Protocol B', 'value': 'B'},
                                {'label': 'Protocol C', 'value': 'C'},
                                {'label': 'Protocol C (Baseline)', 'value': 'baseline_C'}
                            ],
                            value='B',
                            className="mb-3 dash-dropdown"
                        ),
                        
                        html.Div(id={'type': 'topo-fase-container', 'index': 2}, children=[
                            html.Label("Fase", className="control-label"),
                            dcc.Dropdown(id={'type': 'topo-fase-dropdown', 'index': 2}, className="mb-3 dash-dropdown"),
                        ]),
                        
                        html.Div(id={'type': 'topo-group-container', 'index': 2}, children=[
                            html.Label("Group", className="control-label"),
                            dcc.Dropdown(id={'type': 'topo-group-dropdown', 'index': 2}, className="mb-3 dash-dropdown")
                        ]),
                        
                        html.Div(id={'type': 'topo-norm-container', 'index': 2}, children=[
                            dcc.Checklist(
                                id={'type': 'topo-norm-check', 'index': 2},
                                options=[{'label': ' Normalizado', 'value': 'yes'}],
                                value=['yes'],
                                className="mb-3"
                            )
                        ], style={'display': 'none'}),
                        
                        html.Label("Scale", className="control-label"),
                        dcc.RadioItems(
                            id={'type': 'topo-scale-radio', 'index': 2},
                            options=[
                                {'label': ' Linear (psd_mean)', 'value': 'linear'},
                                {'label': ' dB (psd_db_mean)', 'value': 'db'}
                            ],
                            value='db',
                            className="mb-3"
                        )
                    ], className="p-2 border rounded")
                ], open=True, className="mb-3")
            ]),
            
            html.Div(id='topo-controls-3-wrapper', style={'display':'none'}, children=[
                html.Details([
                    html.Summary("🔧 Options - Panel 3", style={'cursor': 'pointer', 'fontWeight': 'bold', 'marginBottom': '10px'}),
                    html.Div([
                        html.Label("Protocol", className="control-label"),
                        dcc.Dropdown(
                            id={'type': 'topo-prot-dropdown', 'index': 3},
                            options=[
                                {'label': 'Protocol A', 'value': 'A'},
                                {'label': 'Protocol B', 'value': 'B'},
                                {'label': 'Protocol C', 'value': 'C'},
                                {'label': 'Protocol C (Baseline)', 'value': 'baseline_C'}
                            ],
                            value='C',
                            className="mb-3 dash-dropdown"
                        ),
                        
                        html.Div(id={'type': 'topo-fase-container', 'index': 3}, children=[
                            html.Label("Fase", className="control-label"),
                            dcc.Dropdown(id={'type': 'topo-fase-dropdown', 'index': 3}, className="mb-3 dash-dropdown"),
                        ]),
                        
                        html.Div(id={'type': 'topo-group-container', 'index': 3}, children=[
                            html.Label("Group", className="control-label"),
                            dcc.Dropdown(id={'type': 'topo-group-dropdown', 'index': 3}, className="mb-3 dash-dropdown")
                        ]),
                        
                        html.Div(id={'type': 'topo-norm-container', 'index': 3}, children=[
                            dcc.Checklist(
                                id={'type': 'topo-norm-check', 'index': 3},
                                options=[{'label': ' Normalizado', 'value': 'yes'}],
                                value=['yes'],
                                className="mb-3"
                            )
                        ], style={'display': 'none'}),
                        
                        html.Label("Scale", className="control-label"),
                        dcc.RadioItems(
                            id={'type': 'topo-scale-radio', 'index': 3},
                            options=[
                                {'label': ' Linear (psd_mean)', 'value': 'linear'},
                                {'label': ' dB (psd_db_mean)', 'value': 'db'}
                            ],
                            value='db',
                            className="mb-3"
                        )
                    ], className="p-2 border rounded")
                ], open=True, className="mb-3")
            ]),

            html.Label("Escala do Mapa", className="control-label"),
            dcc.Dropdown(
                id='topo-scale-mode-dropdown',
                options=[
                    {'label': 'Global (Todos os Protocolos)', 'value': 'global'},
                    {'label': 'Por Protocolo', 'value': 'protocol'},
                    {'label': 'Por Protocolo e Fase', 'value': 'context'},
                    {'label': 'Independente (Por Banda)', 'value': 'independent'}
                ],
                value='global',
                className="mb-3 dash-dropdown"
            ),

            html.Div(id='topo-compare-mode-wrapper', style={'display': 'none'}, children=[
                dcc.Checklist(
                    id='topo-compare-mode-check',
                    options=[{'label': ' Modo Comparação (inverter linhas por colunas)', 'value': 'yes'}],
                    value=[],
                    className="mb-3"
                )
            ]),

            html.Button("Run Topoplot", id='run-topo-btn', className="btn btn-primary w-100"),
            html.Hr(),
            
            # --- Channel Reference Map Card ---
            html.Div([
                html.Details([
                    html.Summary("📍 Channel Reference Map", style={'cursor': 'pointer', 'fontWeight': 'bold'}),
                    html.Div([
                        html.P("Mapping of the 32 channels used (Standard 10-20)", className="text-muted small mb-2"),
                        html.Img(src=f"data:image/png;base64,{get_channel_reference_base64()}", style={'width': '100%', 'height': 'auto'})
                    ], className="text-center p-2")
                ], open=False)
            ], className="card p-2 mt-3 bg-light shadow-sm"),

            html.Div(id='topo-info-panel', className="card p-3 mt-3 text-muted", style={'fontSize': '0.9em'})
        ], className="sidebar"),
        
        html.Div([
            html.Div([
                html.Div([
                    html.Button("ℹ️ Quick Guide", id={'type': 'guide-btn', 'index': 'topo'}, className='btn btn-sm btn-outline-info'),
                    html.Div(id="topo-download-buttons-container", className="d-flex align-items-center gap-2", style={'display': 'inline-flex'})
                ], style={'position': 'absolute', 'top': '10px', 'left': '15px', 'zIndex': 500, 'display': 'flex', 'alignItems': 'center', 'gap': '10px'}),
                html.Div(id="topo-output-container", className="d-flex flex-column gap-3 w-100", style={'paddingTop': '40px'})
            ], className="card p-3 mb-3", style={'position': 'relative'})
        ], className="main-content")
    ])


def get_performance_layout():
    return html.Div([
        html.Div([
            html.H2("EEG PSD Dashboard", className="text-primary mb-4"),
            html.H5("v2 Performance Evaluation", className="text-muted mb-4"),
            html.Hr(),
            
            html.Label("Protocol", className="control-label"),
            dcc.Dropdown(
                id='perf-protocol-dropdown',
                options=[
                    {'label': 'Protocol A', 'value': 'A'},
                    {'label': 'Protocol B', 'value': 'B'},
                    {'label': 'Protocol C', 'value': 'C'}
                ],
                value='A',
                className="mb-3 dash-dropdown"
            ),
            
            html.Hr(),
            
            dcc.Checklist(
                id='perf-normality-check',
                options=[{'label': ' Testar Normalidade', 'value': 'yes'}],
                value=[],
                className="mb-2"
            ),
            html.Div(id='perf-normality-controls', style={'display': 'none'}, children=[
                html.Label("Variável de Teste", className="control-label"),
                dcc.Dropdown(id='perf-normality-var', options=Y_VARIABLES, value='Desempenho', className="mb-2 dash-dropdown"),
                html.Div(id='perf-normality-group-container', children=[
                    html.Label("Grupo", className="control-label"),
                    dcc.Dropdown(id='perf-normality-group', options=[{'label': 'CV', 'value': 'CV'}, {'label': 'SV', 'value': 'SV'}, {'label': 'Ambos', 'value': 'Ambos'}], value='Ambos', className="mb-2 dash-dropdown")
                ])
            ]),
            
            html.Hr(),
            
            dcc.Checklist(
                id='perf-anova-check',
                options=[{'label': ' Executar ANOVA Typ=3 & Post-Hoc', 'value': 'yes'}],
                value=[],
                className="mb-2"
            ),
            html.Div(id='perf-anova-controls', style={'display': 'none'}, children=[
                html.Label("Variável Dependente (Y)", className="control-label"),
                dcc.Dropdown(id='perf-anova-target', options=Y_VARIABLES, value='Desempenho', className="mb-2 dash-dropdown"),
                html.Label("Variáveis Independentes (X)", className="control-label"),
                dcc.Checklist(id='perf-anova-independents', options=[
                    {'label': 'Complexidade', 'value': 'Complexidade'},
                    {'label': 'Grupo', 'value': 'grupo'},
                    {'label': 'Overlap', 'value': 'Overlap'}
                ], value=['Complexidade'], className="mb-2 list-style-none", labelStyle={'display': 'block', 'marginBottom': '5px'}),
                dcc.Checklist(id='perf-heatmap-check', options=[{'label': ' Exibir Heatmap de Significância', 'value': 'yes'}], value=[], className="mt-2 mb-2")
            ]),
            
            html.Hr(),
            
            html.Div(id='perf-alpha-container', style={'display': 'none'}, children=[
                html.Label("Nível de Significância (Alpha)", className="control-label"),
                dcc.Input(id='perf-alpha-input', type='number', value=0.05, min=0.001, max=0.1, step=0.01, className="form-control mb-3")
            ]),

            html.Div(id='perf-interaction-check-container', children=[
                dcc.Checklist(
                    id='perf-interaction-check',
                    options=[{'label': ' Plotar Interaction Plot', 'value': 'yes'}],
                    value=[],
                    className="mb-2"
                ),
            ]),
            dcc.Checklist(
                id='perf-hybrid-check',
                options=[{'label': ' Plotar Gráfico Híbrido (Dispersão + Interação)', 'value': 'yes'}],
                value=[],
                className="mb-2"
            ),
            html.Div(id='perf-interaction-controls', style={'display': 'none'}, children=[
                html.Label("Eixo X", className="control-label"),
                dcc.Dropdown(id='perf-interaction-x', options=[], className="mb-2 dash-dropdown"),
                html.Label("Eixo Y", className="control-label"),
                dcc.Dropdown(id='perf-interaction-y', options=Y_VARIABLES, value='Desempenho', className="mb-2 dash-dropdown"),
                html.Label("Linha", className="control-label"),
                dcc.Dropdown(id='perf-interaction-line', options=[], className="mb-2 dash-dropdown"),
                html.Label("Facet", className="control-label"),
                dcc.Dropdown(id='perf-interaction-facet', options=[{'label': 'None', 'value': 'None'}], value='None', className="mb-2 dash-dropdown"),
                html.Label("Escala Y (ex: 0.2, 1.0)", className="control-label"),
                dcc.Input(id='perf-interaction-ylim', type='text', value='(0.2, 1.0)', className="form-control mb-2")
            ]),
            
            html.Hr(),
            html.Button("Executar Análise", id='perf-run-btn', className="btn btn-primary w-100 mt-2"),
            html.Div(id='perf-controls-error', className="text-danger mt-2")
        ], className="sidebar"),
        
        html.Div([
            html.Div(id='perf-normality-result-card', style={'display': 'none'}, className="card p-3 mb-3 shadow-sm", children=[
                html.H4("Teste de Normalidade", className="text-info"),
                html.Div(id='perf-normality-output')
            ]),
            
            html.Div(id='perf-anova-result-card', style={'display': 'none'}, className="card p-3 mb-3 shadow-sm", children=[
                html.H4("ANOVA e Post-Hoc", className="text-warning"),
                html.Div(id='perf-anova-table-output', style={'overflowX': 'auto'}),
                dcc.Graph(id='perf-anova-dotplot', style={'height': '600px', 'marginTop': '15px'}),
                html.Div(id='perf-anova-heatmap-container', style={'display': 'none'}, children=[
                    html.H5("Heatmap de Significância (Tukey HSD)", className="text-primary mt-3"),
                    dcc.Graph(id='perf-anova-heatmap', style={'height': '500px', 'marginTop': '10px'})
                ]),
                html.Details([
                    html.Summary("Tabela de Resultados Post-Hoc (Significativos)", style={'cursor': 'pointer', 'fontWeight': 'bold', 'color': '#0d6efd', 'marginTop': '15px'}),
                    html.Div(id='perf-anova-posthoc-output', className="mt-2 bg-light p-2 rounded", style={'overflowX': 'auto'})
                ])
            ]),
            
            html.Div(id='perf-interaction-result-card', style={'display': 'none'}, className="card p-3 mb-3 shadow-sm", children=[
                html.H4("Interaction Plot", className="text-success"),
                dcc.Graph(id='perf-interaction-plot', style={'height': '500px'})
            ]),

            html.Div(id='perf-hybrid-result-card', style={'display': 'none'}, className="card p-3 mb-3 shadow-sm", children=[
                html.H4("Gráfico Híbrido (Dispersão + Interação)", className="text-primary"),
                dcc.Graph(id='perf-hybrid-plot', style={'height': '550px'})
            ])
            
        ], className="main-content")
    ])


# --- Layout ---
app.layout = html.Div([
    # Top Header Controls
    html.Div([
        html.Button("🌙", id='theme-toggle', className='theme-toggle', n_clicks=0),
    ], style={'position': 'absolute', 'top': '15px', 'right': '20px', 'zIndex': 1000}),
    
    dcc.Store(id='theme-store', data='light'),
    html.Div(id='theme-injector', style={'display': 'none'}),
    
    # Quick Guide Modal Overlay
    html.Div(id='quick-guide-modal', style={'display': 'none'}, children=[
        html.Div(className='modal-backdrop', style={
            'position': 'fixed', 'top': 0, 'left': 0, 'width': '100vw', 'height': '100vh',
            'backgroundColor': 'rgba(0, 0, 0, 0.7)', 'zIndex': 1040
        }),
        html.Div(className='modal-content-wrapper', style={
            'position': 'fixed', 'top': '5vh', 'left': '10vw', 'width': '80vw', 'height': '90vh',
            'zIndex': 1050, 'backgroundColor': 'inherit', 'borderRadius': '10px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.5)',
            'display': 'flex', 'flexDirection': 'column', 'overflow': 'hidden'
        }, children=[
            html.Div(className='modal-header card-header d-flex justify-content-between align-items-center', style={'padding': '15px'}, children=[
                html.H4("📖 Guia Rápido e Metodológico", className='m-0'),
                html.Button("✖ Fechar", id='close-guide-btn', className='btn btn-danger btn-sm')
            ]),
            html.Div(className='modal-body card-body', style={'overflowY': 'auto', 'padding': '30px'}, children=[
                dcc.Markdown(quick_guide_content, mathjax=True)
            ])
        ])
    ]),
    
    html.Div([
        html.Div([
            dcc.Tabs(id='app-tabs', value='tab-analysis', children=[
                dcc.Tab(label='Multivariate Analysis', value='tab-analysis', className='custom-tab', selected_className='custom-tab--selected'),
                dcc.Tab(label='PSD Visualization', value='tab-psd', className='custom-tab', selected_className='custom-tab--selected'),
                dcc.Tab(label='Topoplot', value='tab-topoplot', className='custom-tab', selected_className='custom-tab--selected'),
                dcc.Tab(label='Performance Evaluation', value='tab-performance', className='custom-tab', selected_className='custom-tab--selected')
            ], className='custom-tabs-container')
        ], className='main-content', style={'paddingTop': '0px', 'paddingBottom': '0px'}),
        html.Div(id='tabs-content')
    ], style={'paddingTop': '40px'})
])

# --- Callbacks ---

@app.callback(
    Output('tabs-content', 'children'),
    [Input('app-tabs', 'value')]
)
def render_content(tab):
    if tab == 'tab-analysis':
        return get_analysis_layout()
    elif tab == 'tab-psd':
        return get_psd_layout()
    elif tab == 'tab-performance':
        return get_performance_layout()
    elif tab == 'tab-topoplot':
        return get_topoplot_layout()

@app.callback(
    [Output('quick-guide-modal', 'style'),
     Output('quick-guide-modal', 'children')],
    [Input({'type': 'guide-btn', 'index': ALL}, 'n_clicks'),
     Input('close-guide-btn', 'n_clicks')],
    State('quick-guide-modal', 'style'),
    prevent_initial_call=True
)
def toggle_quick_guide(btn_clicks, close_clicks, current_style):
    from dash import ctx
    if not ctx.triggered:
        return {'display': 'none'}, dash.no_update
        
    # Validation: Ensure it was a real click, not just a layout update
    trigger = ctx.triggered[0]
    if trigger['value'] is None or trigger['value'] == 0:
        return {'display': 'none'}, dash.no_update
        
    trigger_id_json = trigger['prop_id'].split('.')[0]
    
    if trigger_id_json == 'close-guide-btn':
        return {'display': 'none'}, dash.no_update
    
    # Check if any guide button was actually clicked (not just added to layout)
    # trigger_id_json will be like '{"index":"topo","type":"guide-btn"}'
    import json
    try:
        tid = json.loads(trigger_id_json)
        mode = tid.get('index', '')
    except:
        return {'display': 'none'}, dash.no_update

    # Select content based on trigger
    if mode == 'topo':
        try:
            with open(os.path.join(base_path, 'TOPO_GUIDE.md'), 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            content = topo_guide_content
        title = "📖 Guia Rápido: Topoplots"
    else:
        try:
            with open(os.path.join(base_path, 'QUICK_GUIDE.md'), 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            content = quick_guide_content
        title = "📖 Guia Rápido e Metodológico"

    # Reconstruct the modal content with the correct markdown
    modal_children = [
        html.Div(className='modal-backdrop', style={
            'position': 'fixed', 'top': 0, 'left': 0, 'width': '100vw', 'height': '100vh',
            'backgroundColor': 'rgba(0, 0, 0, 0.7)', 'zIndex': 1040
        }),
        html.Div(className='modal-content-wrapper', style={
            'position': 'fixed', 'top': '5vh', 'left': '10vw', 'width': '80vw', 'height': '90vh',
            'zIndex': 1050, 'backgroundColor': 'inherit', 'borderRadius': '10px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.5)',
            'display': 'flex', 'flexDirection': 'column', 'overflow': 'hidden'
        }, children=[
            html.Div(className='modal-header card-header d-flex justify-content-between align-items-center', style={'padding': '15px'}, children=[
                html.H4(title, className='m-0'),
                html.Button("✖ Fechar", id='close-guide-btn', className='btn btn-danger btn-sm')
            ]),
            html.Div(className='modal-body card-body', style={'overflowY': 'auto', 'padding': '30px'}, children=[
                dcc.Markdown(content, mathjax=True)
            ])
        ])
    ]
    
    return {'display': 'block'}, modal_children

@app.callback(
    Output('math-model-container', 'children'),
    [Input('protocol-dropdown', 'value'),
     Input({'type': 'x-mode-dropdown', 'index': 1}, 'value'),
     Input({'type': 'fase-dropdown', 'index': 1}, 'value'),
     Input({'type': 'channels-dropdown', 'index': 1}, 'value'),
     Input({'type': 'y-checklist', 'index': 1}, 'value')]
)
def update_math_model(prot, x_mode, fase, channels, y_cols):
    try:
        # Load caching internally so it doesn't freeze the UI
        if prot not in data_cache:
            df, meta = load_data(protocol=prot)
            data_cache[prot] = (df, meta)
        else:
            df, meta = data_cache[prot]
            
        if not y_cols:
            y_cols = ['Desempenho']
            
        if not channels:
            channels = ['CZ', 'C3', 'C4']

        X = build_X(df, x_mode, fase=fase, selected_channels=channels)
        Y = build_Y(df, y_cols)
        
        T = X.shape[0] if not X.empty else "T"
        F = X.shape[1] if not X.empty else "F"
        C = Y.shape[1] if not Y.empty else 1
        
        # Build human-readable X mode description
        x_mode_labels = {
            'psd_full_norm': 'PSD Completo Normalizado (Baseline)',
            'psd_2em2_norm': 'PSD Potência de 2 em 2 Hz Normalizado'
        }
        x_mode_label = x_mode_labels.get(x_mode, x_mode)
        
        # Build X description with frequency resolution info
        if x_mode == 'psd_2em2_norm':
            n_channels = len(channels) if channels else 0
            n_freq_bins = F // n_channels if n_channels > 0 else F
            x_desc = f"Potência PSD normalizada em bins de 2 Hz (0–50 Hz → {n_freq_bins} bins × {n_channels} canais = {F} features)"
        else:
            n_channels = len(channels) if channels else 0
            n_freq_pts = F // n_channels if n_channels > 0 else F
            x_desc = f"PSD completo normalizado ({n_freq_pts} pontos de freq × {n_channels} canais = {F} features)"
        
        math_str = rf'''
$$
X = \begin{{bmatrix}}
x_{{1,1}} & x_{{1,2}} & \dots & x_{{1,{F}}} \\
x_{{2,1}} & x_{{2,2}} & \dots & x_{{2,{F}}} \\
\vdots & \vdots & \ddots & \vdots \\
x_{{{T},1}} & x_{{{T},2}} & \dots & x_{{{T},{F}}}
\end{{bmatrix}} \in \mathbb{{R}}^{{{T} \times {F}}}
$$

$$
Y = \begin{{bmatrix}}
y_{{1,1}} & \dots & y_{{1,{C}}} \\
y_{{2,1}} & \dots & y_{{2,{C}}} \\
\vdots & \ddots & \vdots \\
y_{{{T},1}} & \dots & y_{{{T},{C}}}
\end{{bmatrix}} \in \mathbb{{R}}^{{{T} \times {C}}}
$$

**Onde:**
* **T = {T}**: Número total de *trials* analisados do protocolo {prot}.
* **F = {F}**: Número de *features* em $X$.
* **Modo X**: {x_mode_label}
* **Descrição X**: {x_desc}
* **C = {C}**: Variáveis comportamentais integradas em $Y$.
* $x_{{ij}}$ = valor oriundo no espaço PSD na banda/frequência *j* da execução *i*.
* $y_{{ic}}$ = escore medido na target *c* da execução *i*.
'''
        return dcc.Markdown(math_str, mathjax=True)
    except Exception as e:
        return html.Div(f"Model rendering offline: {str(e)}", className="text-danger")



@app.callback(
    [Output('theme-store', 'data'), Output('theme-toggle', 'children')],
    Input('theme-toggle', 'n_clicks'),
    State('theme-store', 'data')
)
def toggle_theme(n, theme):
    if n == 0:
        return 'light', '🌙'
    new = 'dark' if theme == 'light' else 'light'
    return new, '☀️' if new == 'dark' else '🌙'

app.clientside_callback(
    """
    function(theme) {
        if (theme === 'dark') {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
        }
        return '';
    }
    """,
    Output('theme-injector', 'children'),
    Input('theme-store', 'data'),
    prevent_initial_call=False
)

@app.callback(
    [Output('controls-2-container', 'children'),
     Output('single-view', 'style'),
     Output('comparison-view', 'style')],
    Input('comparison-toggle', 'value')
)
def toggle_comparison(enabled):
    if 'yes' in enabled:
        return create_analysis_controls(2), {'display': 'none'}, {'display': 'block'}
    return html.Div(), {'display': 'block'}, {'display': 'none'}

# Protocol Change -> Group Options & Visibility
@app.callback(
    [Output('group-filter-container', 'style'),
     Output('group-checklist', 'options'),
     Output('group-checklist', 'value')],
    [Input('protocol-dropdown', 'value')]
)
def update_group_options(prot):
    if prot == 'A':
        return {'display': 'block'}, [{'label': 'CV', 'value': 'CV'}, {'label': 'SV', 'value': 'SV'}], ['CV', 'SV']
    elif prot == 'B':
        return {'display': 'block'}, [{'label': 'CF', 'value': 'CF'}, {'label': 'SF', 'value': 'SF'}], ['CF', 'SF']
    else:
        return {'display': 'none'}, [], []

@app.callback(
    [Output({'type': 'supervision-dropdown', 'index': ALL}, 'options'),
     Output({'type': 'supervision-dropdown', 'index': ALL}, 'value')],
    [Input('protocol-dropdown', 'value')],
    [State({'type': 'supervision-dropdown', 'index': ALL}, 'value'),
     State({'type': 'supervision-dropdown', 'index': ALL}, 'id')]
)
def update_supervision_options(prot, current_values, ids):
    if not ids:
        return dash.no_update
    
    if prot == 'A':
        options = [
            {'label': 'Group (CV/SV)', 'value': 'group'},
            {'label': 'Complexity', 'value': 'complexity'},
            {'label': 'Overlap (Prot A)', 'value': 'overlap'},
            {'label': 'Group + Complexity', 'value': 'group_comp'},
            {'label': 'Group + Overlap', 'value': 'group_overlap'},
            {'label': 'Group + Comp + Overlap', 'value': 'all'}
        ]
        default_val = 'group'
    elif prot == 'B':
        options = [
            {'label': 'Group (CF/SF)', 'value': 'group'},
            {'label': 'Complexity', 'value': 'complexity'},
            {'label': 'Group + Complexity', 'value': 'group_comp'}
        ]
        default_val = 'group'
    else:
        options = [
            {'label': 'Complexity', 'value': 'complexity'}
        ]
        default_val = 'complexity'
        
    ret_options = [options for _ in ids]
    ret_values = [default_val if v not in [opt['value'] for opt in options] else v for v in current_values]
    
    return ret_options, ret_values

@app.callback(
    Output({'type': 'supervision-container', 'index': MATCH}, 'style'),
    [Input({'type': 'method-dropdown', 'index': MATCH}, 'value')]
)
def update_supervision_visibility(meth):
    return {'display': 'block'} if meth in ['CDA', 'LDA'] else {'display': 'none'}

# Mixed Axes UI Visibility Callback
@app.callback(
    [Output({'type': 'mixed-axes-container', 'index': ALL}, 'style'),
     Output({'type': 'axis3-container', 'index': ALL}, 'style')],
    [Input({'type': 'domain-dropdown', 'index': ALL}, 'value'),
     Input('global-dimensions-radio', 'value')]
)
def update_axis_selector_visibility(domains, dims):
    visibles_container = [{'display': 'block'} if d == 'both' else {'display': 'none'} for d in domains]
    visibles_3d = [{'display': 'block'} if (d == 'both' and dims == 3) else {'display': 'none'} for d in domains]
    return visibles_container, visibles_3d

# Options Populater for Mixed Axes
@app.callback(
     [Output({'type': 'axis-select', 'index': ALL, 'axis': 1}, 'options'),
      Output({'type': 'axis-select', 'index': ALL, 'axis': 2}, 'options'),
      Output({'type': 'axis-select', 'index': ALL, 'axis': 3}, 'options'),
      Output({'type': 'axis-select', 'index': ALL, 'axis': 1}, 'value'),
      Output({'type': 'axis-select', 'index': ALL, 'axis': 2}, 'value'),
      Output({'type': 'axis-select', 'index': ALL, 'axis': 3}, 'value')],
     [Input('protocol-dropdown', 'value'),
      Input({'type': 'domain-dropdown', 'index': ALL}, 'value')] # simple triggers
)
def options_axis_selectors(protocol, domains):
     # Provide a static, safe set of maximum options since actual component 
     # dimension is deferred to the analysis compute phase.
     # User instruction: "cada eixo permite escolher: C1 de X, C2 de X... C1 de Y..."
     opts = [
         {'label': 'C1 (X)', 'value': 'C1_X'},
         {'label': 'C2 (X)', 'value': 'C2_X'},
         {'label': 'C3 (X)', 'value': 'C3_X'},
         {'label': 'C1 (Y)', 'value': 'C1_Y'},
         {'label': 'C2 (Y)', 'value': 'C2_Y'},
         {'label': 'C3 (Y)', 'value': 'C3_Y'},
     ]
     
     opt1 = [opts for _ in domains]
     opt2 = [opts for _ in domains]
     opt3 = [opts for _ in domains]
     
     val1 = ['C1_X' for _ in domains]
     val2 = ['C2_X' for _ in domains]
     val3 = ['C1_Y' for _ in domains]

     return opt1, opt2, opt3, val1, val2, val3

@app.callback(
    [Output('plot-1', 'figure'), Output('stats-1', 'children'),
     Output('anova-plot-1', 'figure'), Output('anova-stats-1', 'children'),
     Output('centroid-distance-1', 'children'),
     Output('info-panel', 'children')],
    Input('run-btn', 'n_clicks'),
    [State('protocol-dropdown', 'value'),
     State('group-checklist', 'value'),
     State({'type': 'method-dropdown', 'index': 1}, 'value'),
     State({'type': 'x-mode-dropdown', 'index': 1}, 'value'),
     State({'type': 'fase-dropdown', 'index': 1}, 'value'),
     State({'type': 'channels-dropdown', 'index': 1}, 'value'),
     State({'type': 'y-checklist', 'index': 1}, 'value'),
     State({'type': 'domain-dropdown', 'index': 1}, 'value'),
     State('global-dimensions-radio', 'value'),
     State({'type': 'color-dropdown', 'index': 1}, 'value'),
     State({'type': 'supervision-dropdown', 'index': 1}, 'value'),
     State('theme-store', 'data'),
     State({'type': 'axis-select', 'index': 1, 'axis': 1}, 'value'),
     State({'type': 'axis-select', 'index': 1, 'axis': 2}, 'value'),
     State({'type': 'axis-select', 'index': 1, 'axis': 3}, 'value'),
     State('comparison-toggle', 'value')],
    prevent_initial_call=True
)
def update_single_analysis(n, prot, groups, meth, x_mode, fase, selected_channels, y_cols, dom, dims, color, supervision_by, theme, ax1, ax2, ax3, comp):
    if n == 0 or 'yes' in comp:
        fig = go.Figure()
        fig.update_layout(title="Click Run Analysis")
        return fig, "No data", go.Figure(), "", html.P("Ready")
    
    axes = [ax1, ax2, ax3]
    fig, stats, anova_fig, anova_res, centroid_res = run_single_analysis(prot, groups, meth, x_mode, y_cols, dom, axes, dims, theme, color, supervision_by, fase, selected_channels)
    
    info = html.Div([
        html.P([html.Strong("Protocol: "), prot]),
        html.P([html.Strong("Groups: "), ", ".join(groups) if groups else "All"]),
        html.H6(f"Method: {meth}", style={'marginTop': '10px'}),
    ])
    return fig, stats, anova_fig, anova_res, centroid_res, info

@app.callback(
    [Output('plot-left', 'figure'), Output('stats-left', 'children'),
     Output('anova-plot-left', 'figure'), Output('anova-stats-left', 'children'),
     Output('centroid-distance-left', 'children'),
     Output('plot-right', 'figure'), Output('stats-right', 'children'),
     Output('anova-plot-right', 'figure'), Output('anova-stats-right', 'children'),
     Output('centroid-distance-right', 'children')],
    Input('run-btn', 'n_clicks'),
    [State('protocol-dropdown', 'value'),
     State('group-checklist', 'value'),
     State({'type': 'method-dropdown', 'index': ALL}, 'value'),
     State({'type': 'x-mode-dropdown', 'index': ALL}, 'value'),
     State({'type': 'fase-dropdown', 'index': ALL}, 'value'),
     State({'type': 'channels-dropdown', 'index': ALL}, 'value'),
     State({'type': 'y-checklist', 'index': ALL}, 'value'),
     State({'type': 'domain-dropdown', 'index': ALL}, 'value'),
     State('global-dimensions-radio', 'value'),
     State({'type': 'color-dropdown', 'index': ALL}, 'value'),
     State({'type': 'supervision-dropdown', 'index': ALL}, 'value'),
     State('theme-store', 'data'),
     State('comparison-toggle', 'value'),
     State({'type': 'axis-select', 'index': ALL, 'axis': 1}, 'value'),
     State({'type': 'axis-select', 'index': ALL, 'axis': 2}, 'value'),
     State({'type': 'axis-select', 'index': ALL, 'axis': 3}, 'value')],
    prevent_initial_call=True
)
def update_comparison(n, prot, groups, methods, x_modes, fases, selected_channels_lists, y_cols_lists, doms, dims, colors, supervisions, theme, comp, ax1s, ax2s, ax3s):
    fig = go.Figure()
    fig.update_layout(title="Enable comparison mode")
    
    if n == 0 or 'yes' not in comp or len(methods) < 2:
        return fig, "Waiting...", go.Figure(), "", "", fig, "Waiting...", go.Figure(), "", ""
        
    # Analysis 1
    axes1 = [ax1s[0], ax2s[0], ax3s[0]]
    fig1, stats1, anova_fig1, anova_res1, centroid_res1 = run_single_analysis(
        prot, groups, methods[0], x_modes[0], y_cols_lists[0], doms[0], axes1, dims, theme, colors[0], supervisions[0], fases[0], selected_channels_lists[0]
    )
    
    # Analysis 2
    axes2 = [ax1s[1], ax2s[1], ax3s[1]]
    fig2, stats2, anova_fig2, anova_res2, centroid_res2 = run_single_analysis(
        prot, groups, methods[1], x_modes[1], y_cols_lists[1], doms[1], axes2, dims, theme, colors[1], supervisions[1], fases[1], selected_channels_lists[1]
    )
    
    return fig1, stats1, anova_fig1, anova_res1, centroid_res1, fig2, stats2, anova_fig2, anova_res2, centroid_res2

# Expose server for Vercel
server = app.server

@app.callback(
    [Output('psd-stratify-dropdown', 'options'),
     Output('psd-stratify-dropdown', 'value')],
    [Input('psd-protocol-dropdown', 'value')]
)
def update_psd_stratify_options(prot):
    if prot == 'A':
        options = [
            {'label': 'Group (CV/SV)', 'value': 'grupo'},
            {'label': 'Complexity', 'value': 'complexity'},
            {'label': 'Overlap (Prot A)', 'value': 'overlap'}
        ]
        return options, 'grupo'
    elif prot == 'B':
        options = [
            {'label': 'Group (CF/SF)', 'value': 'grupo'},
            {'label': 'Complexity', 'value': 'complexity'}
        ]
        return options, 'grupo'
    else:
        options = [
            {'label': 'Complexity', 'value': 'complexity'}
        ]
        return options, 'complexity'


@app.callback(
    [Output('psd-main-plot', 'figure'),
     Output('psd-info-panel', 'children')],
    [Input('run-psd-btn', 'n_clicks')],
    [State('psd-protocol-dropdown', 'value'),
     State('psd-fase-dropdown', 'value'),
     State('psd-channels-checklist', 'value'),
     State('psd-scale-radio', 'value'),
     State('psd-bands-toggle', 'value'),
     State('psd-stratify-dropdown', 'value'),
     State('psd-overlay-toggle', 'value'),
     State('theme-store', 'data')],
    prevent_initial_call=True
)
def run_psd_visualization(n_clicks, protocol, fase, channels, scale, bands_toggle, stratify_by, overlay_toggle, theme):
    try:
        # Load caching
        if protocol not in data_cache:
            df, meta = load_data(protocol=protocol)
            data_cache[protocol] = (df, meta)
        else:
            df, meta = data_cache[protocol]
            
        # The PSD visualizing engine relies on the protX_x_psd_norm.csv raw rows without dropping groups
        # (Though groups are handled gracefully later, it's better to provide the pure data and strata)
        # However, data_loader.load_data already dropped some bad rows. So we get build_X to load perfectly aligned rows.
        df_x = build_X(df, 'psd_full_norm', fase=fase, selected_channels=channels)
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"Error Loading PSD Data: {str(e)}")
        return fig, html.Div(f"Data Loader Exception: {str(e)}", className="text-danger")

    # Map the dropdown target names to the actual metadata columns
    stratify_col = None
    if stratify_by == 'grupo':
        stratify_col = 'grupo'
    elif stratify_by == 'complexity':
        stratify_col = 'Complexidade'
    elif stratify_by == 'overlap':
        stratify_col = 'Overlap'

    show_bands = 'yes' in (bands_toggle or [])
    overlay_strata = 'yes' in (overlay_toggle or [])
    
    try:
        fig = create_psd_subplots(
            df_meta=meta.copy(),
            df_x=df_x,
            channels_selected=channels,
            stratify_by=stratify_col,
            scale=scale,
            show_bands=show_bands,
            overlay_strata=overlay_strata,
            theme=theme
        )
        
        info_html = html.Div(
            f"Successfully rendered {len(channels)} channels × {len(meta[stratify_col].unique()) if stratify_col in meta.columns else 1} conditions.", 
            className="text-success"
        )
        return fig, info_html
    except Exception as e:
        import traceback
        traceback.print_exc()
        fig = go.Figure()
        fig.update_layout(title=f"Rendering Error: {str(e)}")
        return fig, html.Div(f"Plotting Exception: {str(e)}", className="text-danger")


# --- Channel Selection Callbacks ---

# Combined callback to handle both protocol-based filtering and "Select All" logic
@app.callback(
    [Output({'type': 'channels-dropdown', 'index': ALL}, 'options'),
     Output({'type': 'channels-dropdown', 'index': ALL}, 'value')],
    [Input('protocol-dropdown', 'value'),
     Input({'type': 'channels-dropdown', 'index': ALL}, 'value')],
    [State({'type': 'channels-dropdown', 'index': ALL}, 'id')],
    prevent_initial_call=False
)
def update_channels_multivariate(prot, current_values, ids):
    if not ids:
        return dash.no_update
        
    from dash import ctx
    trigger = ctx.triggered_id if not isinstance(ctx.triggered_id, dict) else ctx.triggered_id.get('type')
    
    channels = CHANNELS_C if prot == 'C' else ALL_CHANNELS
    options = [{'label': 'Select All', 'value': 'ALL'}] + [{'label': ch, 'value': ch} for ch in channels]
    
    ret_options = [options for _ in ids]
    ret_values = []
    
    for i, val in enumerate(current_values):
        if not val:
            ret_values.append(['CZ', 'C3', 'C4'])
            continue
            
        # Handle Select All trigger
        if 'ALL' in val:
            ret_values.append(channels)
        else:
            # Filter by protocol
            ret_values.append([v for v in val if v in channels])
            
    return ret_options, ret_values

@app.callback(
    [Output('psd-channels-checklist', 'options'),
     Output('psd-channels-checklist', 'value')],
    [Input('psd-protocol-dropdown', 'value'),
     Input('psd-channels-checklist', 'value')],
    prevent_initial_call=False
)
def update_channels_psd(prot, current_value):
    channels = CHANNELS_C if prot == 'C' else ALL_CHANNELS
    options = [{'label': 'Select All', 'value': 'ALL'}] + [{'label': ch, 'value': ch} for ch in channels]
    
    if not current_value:
        return options, ['CZ', 'C3', 'C4']
        
    if 'ALL' in current_value:
        return options, channels
    
    new_value = [v for v in current_value if v in channels]
    if not new_value:
        new_value = ['CZ', 'C3', 'C4']
        
    return options, new_value

# --- Fase Options Population Callbacks ---
@app.callback(
    [Output({'type': 'fase-dropdown', 'index': MATCH}, 'options'),
     Output({'type': 'fase-dropdown', 'index': MATCH}, 'value')],
    [Input('protocol-dropdown', 'value')]
)
def update_fase_options_analysis(prot):
    if prot == 'C':
        opts = [{'label': 'Exploração', 'value': 'estimulacao'}, {'label': 'Execução', 'value': 'execucao'}]
    else:
        opts = [{'label': 'Estimulação', 'value': 'estimulacao'}, {'label': 'Execução', 'value': 'execucao'}]
    return opts, 'estimulacao'

@app.callback(
    [Output('psd-fase-dropdown', 'options'),
     Output('psd-fase-dropdown', 'value')],
    [Input('psd-protocol-dropdown', 'value')]
)
def update_fase_options_psd(prot):
    if prot == 'C':
        opts = [{'label': 'Exploração', 'value': 'estimulacao'}, {'label': 'Execução', 'value': 'execucao'}]
    else:
        opts = [{'label': 'Estimulação', 'value': 'estimulacao'}, {'label': 'Execução', 'value': 'execucao'}]
    return opts, 'estimulacao'

# --- Topoplot Callbacks Consolidado ---
@app.callback(
    [Output({'type': 'topo-group-container', 'index': MATCH}, 'style'),
     Output({'type': 'topo-group-dropdown', 'index': MATCH}, 'options'),
     Output({'type': 'topo-group-dropdown', 'index': MATCH}, 'value'),
     Output({'type': 'topo-norm-container', 'index': MATCH}, 'style'),
     Output({'type': 'topo-fase-container', 'index': MATCH}, 'style'),
     Output({'type': 'topo-fase-dropdown', 'index': MATCH}, 'options'),
     Output({'type': 'topo-fase-dropdown', 'index': MATCH}, 'value')],
    [Input({'type': 'topo-prot-dropdown', 'index': MATCH}, 'value')]
)
def update_topo_protocol_options(prot):
    if not prot:
        return {'display': 'none'}, [], None, {'display': 'none'}, {'display': 'none'}, [], None

    style_group = {'display': 'none'}
    group_opts = []
    group_val = None
    style_norm = {'display': 'none'}
    style_fase = {'display': 'block'}
    
    # Fase options logic (A/B share same, C differs)
    if prot == 'C':
        fase_opts = [
            {'label': 'Exploração', 'value': 'estimulacao'}, 
            {'label': 'Execução', 'value': 'execucao'},
            {'label': 'Ambos', 'value': 'Ambos'}
        ]
    else:
        fase_opts = [
            {'label': 'Estimulação', 'value': 'estimulacao'}, 
            {'label': 'Execução', 'value': 'execucao'},
            {'label': 'Ambos', 'value': 'Ambos'}
        ]
    
    fase_val = 'estimulacao'
    
    if prot == 'A':
        style_group = {'display': 'block'}
        group_opts = [
            {'label': 'CV', 'value': 'CV'}, 
            {'label': 'SV', 'value': 'SV'},
            {'label': 'Ambos', 'value': 'Ambos'}
        ]
        group_val = 'CV'
    elif prot == 'B':
        style_group = {'display': 'block'}
        group_opts = [
            {'label': 'CF', 'value': 'CF'}, 
            {'label': 'SF', 'value': 'SF'},
            {'label': 'Ambos', 'value': 'Ambos'}
        ]
        group_val = 'CF'
    elif prot == 'C':
        style_norm = {'display': 'block'}
    elif prot == 'baseline_C':
        style_fase = {'display': 'none'}
        fase_val = None # Not used
        
    return style_group, group_opts, group_val, style_norm, style_fase, fase_opts, fase_val

@app.callback(
    [Output('topo-scale-mode-dropdown', 'options'),
     Output('topo-scale-mode-dropdown', 'value')],
    [Input({'type': 'topo-prot-dropdown', 'index': ALL}, 'value'),
     Input('topo-plot-count-radio', 'value')],
    State('topo-scale-mode-dropdown', 'value')
)
def update_topo_scale_options(prots, plot_count, current_val):
    """
    If 1 plot: Original logic (Global, Protocol, Context, etc.)
    If >1 plot: Simple logic (Global per Band, Global Absolute)
    """
    if not plot_count:
        plot_count = 1

    if plot_count > 1:
        options = [
            {'label': 'Escala por banda (Global por Banda)', 'value': 'global_per_band'},
            {'label': 'Escala comum (Global Absoluto)', 'value': 'global_absolute'}
        ]
        # Ensure current_val is valid for the new options
        if current_val not in ['global_per_band', 'global_absolute']:
            current_val = 'global_per_band'
        return options, current_val

    # Original logic for 1 plot
    active_prots = prots[:plot_count]
    has_baseline = any(p == 'baseline_C' for p in active_prots)
    has_ab = any(p in ['A', 'B'] for p in active_prots)
    
    base_options = [
        {'label': 'Global (Todos os Protocolos)', 'value': 'global'},
        {'label': 'Por Protocolo', 'value': 'protocol'},
        {'label': 'Por Protocolo e Fase', 'value': 'context'},
        {'label': 'Independente (Por Banda)', 'value': 'independent'}
    ]
    
    if has_ab:
        base_options.insert(3, {'label': 'Por Protocolo, Fase e Grupo', 'value': 'group_context'})
    
    if has_baseline:
        base_options = [opt for opt in base_options if opt['value'] not in ['context', 'group_context']]
        if current_val in ['context', 'group_context']:
            current_val = 'global'
            
    return base_options, current_val

@app.callback(
    [Output('topo-controls-2-wrapper', 'style'),
     Output('topo-controls-3-wrapper', 'style'),
     Output('topo-compare-mode-wrapper', 'style')],
    Input('topo-plot-count-radio', 'value')
)
def toggle_topo_panels(count):
    style2 = {'display': 'block'} if count >= 2 else {'display': 'none'}
    style3 = {'display': 'block'} if count >= 3 else {'display': 'none'}
    style_compare = {'display': 'block'} if count >= 2 else {'display': 'none'}
    return style2, style3, style_compare

@app.callback(
    [Output('topo-output-container', 'children'),
     Output('topo-info-panel', 'children'),
     Output('topo-download-buttons-container', 'children')],
    [Input('run-topo-btn', 'n_clicks')],
    [State({'type': 'topo-prot-dropdown', 'index': ALL}, 'value'),
     State({'type': 'topo-fase-dropdown', 'index': ALL}, 'value'),
     State({'type': 'topo-group-dropdown', 'index': ALL}, 'value'),
     State({'type': 'topo-scale-radio', 'index': ALL}, 'value'),
     State({'type': 'topo-norm-check', 'index': ALL}, 'value'),
     State('topo-plot-count-radio', 'value'),
     State('topo-scale-mode-dropdown', 'value'),
     State('topo-compare-mode-check', 'value')],
    prevent_initial_call=True
)
def run_topoplots(n_clicks, prots, fases, groups, scales, norms, plot_count, scale_mode, compare_mode):
    if not n_clicks:
        from dash.exceptions import PreventUpdate
        raise PreventUpdate

    from topoplot_engine import (
        generate_topoplot_grid_base64, 
        generate_topoplot_comparison_base64,
        generate_anova_map_base64,
        get_topoplot_bounds,
        get_topoplot_path,
        generate_inverted_topoplot_grid_base64
    )
        
    
    outputs = []
    messages = []
    download_buttons = []
    
    panels = plot_count if plot_count else 1
    
    # --- Pre-calculate Bounds for all panels ---
    panel_limits = [] # Used for single-plot or global_absolute
    band_limits = None  # Used for global_per_band
    common_vmin, common_vmax = None, None

    if panels > 1:
        # Multi-plot logic: Calculate global limits across all active panels
        all_dfs = []
        target_cols = []
        from topoplot_engine import get_topoplot_path, BANDS_ORDER
        
        for i in range(panels):
            is_baseline = (prots[i] == 'baseline_C')
            fases_to_load = ['estimulacao', 'execucao'] if fases[i] == 'Ambos' else [fases[i]]
            
            for f in fases_to_load:
                fpath = get_topoplot_path('C' if is_baseline else prots[i], f, 'yes' in (norms[i] or []), is_baseline)
                if fpath and os.path.exists(fpath):
                    try:
                        df_tmp = pd.read_csv(fpath)
                        if 'banda' in df_tmp.columns:
                             df_tmp['banda'] = df_tmp['banda'].str.lower().replace({'alpha': 'alfa'})
                        # Filter group if needed (if group is 'Ambos', we keep all and combine later, 
                        # but for bound calculation we can just take all rows of the file)
                        if prots[i] in ['A', 'B'] and not is_baseline and groups[i] != 'Ambos':
                            df_tmp = df_tmp[df_tmp['grupo'] == groups[i]]
                        
                        all_dfs.append(df_tmp)
                        target_cols.append('psd_db_mean' if scales[i] == 'db' else 'psd_mean')
                    except: pass
        
        if scale_mode == 'global_per_band':
            band_limits = {}
            for band in BANDS_ORDER:
                b_mins, b_maxs = [], []
                for df_tmp, tcol in zip(all_dfs, target_cols):
                    df_b = df_tmp[df_tmp['banda'] == band]
                    if not df_b.empty:
                        b_mins.append(df_b[tcol].min())
                        b_maxs.append(df_b[tcol].max())
                if b_mins:
                    bvmin, bvmax = min(b_mins), max(b_maxs)
                    vrange = bvmax - bvmin if bvmax > bvmin else 1.0
                    band_limits[band] = (bvmin - 0.05*vrange, bvmax + 0.05*vrange)
        
        elif scale_mode == 'global_absolute':
            g_mins, g_maxs = [], []
            for df_tmp, tcol in zip(all_dfs, target_cols):
                if not df_tmp.empty:
                    g_mins.append(df_tmp[tcol].min())
                    g_maxs.append(df_tmp[tcol].max())
            if g_mins:
                common_vmin, common_vmax = min(g_mins), max(g_maxs)
                vrange = common_vmax - common_vmin if common_vmax > common_vmin else 1.0
                common_vmin -= 0.05 * vrange
                common_vmax += 0.05 * vrange

    else:
        # Single-plot logic: Original contextual options
        prot = prots[0]
        fase = fases[0]
        scale_db = scales[0] == 'db'
        is_norm = 'yes' in (norms[0] or [])
        is_baseline = (prot == 'baseline_C')
        real_prot = 'C' if is_baseline else prot
        target_col = 'psd_db_mean' if scale_db else 'psd_mean'
        
        vmin, vmax = None, None
        if scale_mode == 'global':
            vmin, vmax = get_topoplot_bounds('global', target_col=target_col)
        elif scale_mode == 'protocol':
            vmin, vmax = get_topoplot_bounds('protocol', protocol=real_prot, target_col=target_col)
        elif scale_mode == 'context' or scale_mode == 'group_context':
            from topoplot_engine import get_topoplot_path
            fpath = get_topoplot_path(real_prot, fase, is_norm, is_baseline)
            if fpath and os.path.exists(fpath):
                try:
                    df_temp = pd.read_csv(fpath)
                    if scale_mode == 'group_context' and 'grupo' in df_temp.columns:
                         df_temp = df_temp[df_temp['grupo'] == groups[0]]
                    if not df_temp.empty:
                        vmin, vmax = df_temp[target_col].min(), df_temp[target_col].max()
                        vrange = vmax - vmin if vmax > vmin else 1.0
                        vmin -= 0.05 * vrange
                        vmax += 0.05 * vrange
                except: pass
        common_vmin, common_vmax = vmin, vmax

    # --- Now Render ---
    is_compare = ('yes' in (compare_mode or [])) and (panels >= 2)
    
    if is_compare:
        panels_data = []
        for i in range(panels):
            prot = prots[i]
            group = groups[i]
            scale_db = scales[i] == 'db'
            is_norm = 'yes' in (norms[i] or [])
            is_baseline = (prot == 'baseline_C')
            real_prot = 'C' if is_baseline else prot
            
            if real_prot == 'A' and group not in ['CV', 'SV', 'Ambos']:
                group = 'CV'
            elif real_prot == 'B' and group not in ['CF', 'SF', 'Ambos']:
                group = 'CF'
                
            panels_data.append({
                'protocol': real_prot,
                'fase': fases[i],
                'group': group,
                'scale_db': scale_db,
                'is_normalized': is_norm,
                'is_baseline': is_baseline
            })
            
        print(f"[DEBUG TOPO] Inverted Comparison | Panels: {panels} | Scale: {scale_mode}")
        
        img_b64, err = generate_inverted_topoplot_grid_base64(
            panels_data=panels_data,
            vmin=common_vmin, vmax=common_vmax, band_limits=band_limits
        )
        
        # Build panel description titles
        title_norm_p1 = " Normalizado" if panels_data[0]['is_normalized'] else ""
        panel_titles = []
        for idx, p in enumerate(panels_data):
            fase_str = p['fase'].capitalize() if p['fase'] else "N/A"
            title_group = f" - Grupo {p['group']}" if p['protocol'] in ['A', 'B'] else ""
            panel_titles.append(f"Painel {idx+1}: {p['protocol']}{title_norm_p1} ({fase_str}{title_group})")
            
        compare_title = "Modo Comparação: " + " VS ".join(panel_titles)
        
        from dash import html
        if err:
            outputs.append(html.Div([
                html.H4(compare_title, className="text-center"),
                html.Div(f"Erro: {err}", className="alert alert-danger mx-auto mt-2", style={'maxWidth': '600px'})
            ], className="card p-3 shadow-sm"))
            messages.append(f"Comparison Grid Failed: {err}")
        else:
            outputs.append(html.Div([
                html.H4(compare_title, className="text-center text-primary mb-2"),
                html.P("Comparação das faixas de frequência (linhas) lado a lado entre os painéis (colunas).", className="text-center text-muted small"),
                html.Img(src=f"data:image/png;base64,{img_b64}", style={'width':'100%', 'height':'auto'})
            ], className="card p-3 shadow-sm"))
            messages.append("Comparison Grid Rendered Successfully.")
            
            # Button next to Quick Guide
            download_buttons.append(html.Div([
                html.Button("📥 Salvar Comparação", id={'type': 'topo-download-btn', 'index': 'compare_grid'}, className="btn btn-sm btn-outline-primary"),
                dcc.Store(id={'type': 'topo-img-store', 'index': 'compare_grid'}, data=img_b64),
                dcc.Download(id={'type': 'topo-download-link', 'index': 'compare_grid'})
            ]))
            
    else:
        for i in range(panels):
            prot = prots[i]
            fase = fases[i]
            group = groups[i]
            scale_db = scales[i] == 'db'
            is_norm = 'yes' in (norms[i] or [])
            is_baseline = (prot == 'baseline_C')
            real_prot = 'C' if is_baseline else prot
            
            # --- Segurança Extra: Forçar grupo/fase correto para o protocolo ---
            if prot == 'A' and group not in ['CV', 'SV', 'Ambos']:
                group = 'CV'
            elif prot == 'B' and group not in ['CF', 'SF', 'Ambos']:
                group = 'CF'
                
            print(f"[DEBUG TOPO] Panel {i+1} | Prot: {prot} | Fase: {fase} | Group: {group} | Scale: {scale_mode}")
            
            img_b64, err = generate_topoplot_grid_base64(
                protocol=real_prot, fase=fase, group=group, 
                scale_db=scale_db, is_normalized=is_norm, is_baseline=is_baseline,
                vmin=common_vmin, vmax=common_vmax, band_limits=band_limits
            )
            
            title_norm = " Normalizado" if is_norm else ""
            title_group = f" - Grupo {group}" if prot in ['A', 'B'] else ""
            title_scale = "(dB)" if scale_db else "(Linear)"
            
            fase_str = "N/A"
            if fase:
                fase_str = fase.capitalize()
                
            panel_title = f"Protocolo {prot}{title_norm} | Fase: {fase_str}{title_group} {title_scale}"
            
            from dash import html
            if err:
                outputs.append(html.Div([
                    html.H4(panel_title, className="text-center"),
                    html.Div(f"Erro: {err}", className="alert alert-danger mx-auto mt-2", style={'maxWidth': '600px'})
                ], className="card p-3 shadow-sm"))
                messages.append(f"Panel {i+1} Failed: {err}")
            else:
                outputs.append(html.Div([
                    html.H4(panel_title, className="text-center text-primary mb-2"),
                    html.Img(src=f"data:image/png;base64,{img_b64}", style={'width':'100%', 'height':'auto'})
                ], className="card p-3 shadow-sm"))
                messages.append(f"Panel {i+1} Rendered Successfully.")
                
                # Button next to Quick Guide
                btn_label = f"📥 Salvar Painel {i+1}" if panels > 1 else "📥 Salvar Topoplot"
                download_buttons.append(html.Div([
                    html.Button(btn_label, id={'type': 'topo-download-btn', 'index': f"panel_{i+1}"}, className="btn btn-sm btn-outline-primary"),
                    dcc.Store(id={'type': 'topo-img-store', 'index': f"panel_{i+1}"}, data=img_b64),
                    dcc.Download(id={'type': 'topo-download-link', 'index': f"panel_{i+1}"})
                ]))
            
    # --- Statistical Comparison Row ---
    if panels >= 2:
        # Comparison logic
        p_list = []
        for i in range(panels):
            p_list.append({
                'protocol': 'C' if (prots[i] == 'baseline_C') else prots[i],
                'fase': fases[i],
                'group': groups[i],
                'is_normalized': 'yes' in (norms[i] or []),
                'is_baseline': (prots[i] == 'baseline_C')
            })

        # 1. Global ANOVA Map (only for 3 panels)
        if panels == 3:
            anova_img, anova_stats = generate_anova_map_base64(p_list, scales[0] == 'db')
            if anova_img:
                anova_details_children = []
                for band, channels in anova_stats.items():
                    if channels:
                        anova_details_children.append(html.B(f"{band.capitalize()}: ", className="text-warning"))
                        anova_details_children.append(html.Span(", ".join(channels)))
                        anova_details_children.append(html.Br())
                
                if not anova_details_children:
                    anova_details_children = [html.I("Nenhum canal sobreviveu à correção FDR (p < 0.05).")]

                outputs.append(html.Div([
                    html.H4("Análise Global (One-Way ANOVA)", className="text-center text-warning mb-2"),
                    html.P("Mapa de probabilidade (1-p). Cores quentes e marcações com 'x' indicam variação significativa entre os 3 painéis (corrigido por FDR).", className="text-center text-muted small"),
                    html.Img(src=f"data:image/png;base64,{anova_img}", style={'width':'100%', 'height':'auto'}),
                    html.Details([
                        html.Summary("📊 Detalhes Estatísticos (ANOVA Global + FDR)", 
                                     style={'cursor': 'pointer', 'fontWeight': 'bold', 'color': '#ffc107', 'marginTop': '10px'}),
                        html.Div(anova_details_children, className="p-2 border rounded bg-light mt-2", style={'fontSize': '0.85em'})
                    ], className="mt-2")
                ], className="card p-3 shadow-sm border-warning", style={'borderWidth': '2px', 'marginTop': '20px'}))
                messages.append("ANOVA Global Rendered.")
                
                # Button next to Quick Guide
                download_buttons.append(html.Div([
                    html.Button("📥 Salvar ANOVA", id={'type': 'topo-download-btn', 'index': 'anova_map'}, className="btn btn-sm btn-outline-warning"),
                    dcc.Store(id={'type': 'topo-img-store', 'index': 'anova_map'}, data=anova_img),
                    dcc.Download(id={'type': 'topo-download-link', 'index': 'anova_map'})
                ]))

        # 2. Pairwise Comparisons
        comps = [(0, 1)]
        if panels == 3: comps = [(0, 1), (0, 2), (1, 2)]
            
        for idx1, idx2 in comps:
            label1, label2 = f"Painel {idx1+1}", f"Painel {idx2+1}"
            img_comp, stats_data = generate_topoplot_comparison_base64(
                p_list[idx1], p_list[idx2], scales[idx1] == 'db',
                label1=label1, label2=label2
            )
            
            if img_comp:
                details_children = []
                for band, channels in stats_data.items():
                    if channels:
                        details_children.append(html.B(f"{band.capitalize()}: ", className="text-primary"))
                        ch_list = ", ".join([f"{c['canal']} (p={c['p']:.3f})" for c in channels])
                        details_children.append(html.Span(ch_list))
                        details_children.append(html.Br())
                
                if not details_children:
                    details_children = [html.I(f"Nenhum canal significativo encontrado (p < 0.05).")]

                outputs.append(html.Div([
                    html.H4(f"Comparação: {label1} vs {label2}", className="text-center text-danger mb-2"),
                    html.Img(src=f"data:image/png;base64,{img_comp}", style={'width':'100%', 'height':'auto'}),
                    html.Details([
                        html.Summary(f"📊 Detalhes Estatísticos ({label1} vs {label2})", 
                                     style={'cursor': 'pointer', 'fontWeight': 'bold', 'color': '#dc3545', 'marginTop': '10px'}),
                        html.Div(details_children, className="p-2 border rounded bg-light mt-2", style={'fontSize': '0.85em'})
                    ], className="mt-2")
                ], className="card p-3 shadow-sm border-danger", style={'borderWidth': '2px', 'marginTop': '20px'}))
                messages.append(f"Comparison {label1} vs {label2} Rendered.")
                
                # Button next to Quick Guide
                download_buttons.append(html.Div([
                    html.Button(f"📥 Salvar Comp. {idx1+1} vs {idx2+1}", id={'type': 'topo-download-btn', 'index': f"comp_{idx1}_{idx2}"}, className="btn btn-sm btn-outline-danger"),
                    dcc.Store(id={'type': 'topo-img-store', 'index': f"comp_{idx1}_{idx2}"}, data=img_comp),
                    dcc.Download(id={'type': 'topo-download-link', 'index': f"comp_{idx1}_{idx2}"})
                ]))

    return outputs, " | ".join(messages), download_buttons

@app.callback(
    Output({'type': 'topo-download-link', 'index': MATCH}, 'data'),
    Input({'type': 'topo-download-btn', 'index': MATCH}, 'n_clicks'),
    State({'type': 'topo-img-store', 'index': MATCH}, 'data'),
    prevent_initial_call=True
)
def download_topo_image(n_clicks, img_base64):
    if not n_clicks or not img_base64:
        from dash.exceptions import PreventUpdate
        raise PreventUpdate
        
    import base64
    from dash import dcc
    img_bytes = base64.b64decode(img_base64)
    
    # Determine which button triggered the callback
    ctx = dash.callback_context
    filename = "topoplot.png"
    if ctx.triggered:
        prop_id = ctx.triggered[0]['prop_id']
        import json
        try:
            trigger_dict = json.loads(prop_id.split('.')[0])
            idx = trigger_dict.get('index', 'topoplot')
            filename = f"topoplot_{idx}.png"
        except:
            pass
            
    return dcc.send_bytes(img_bytes, filename=filename)


# ==============================================================================
# PERFORMANCE EVALUATION CALLBACKS
# ==============================================================================

@app.callback(
    [Output('perf-normality-controls', 'style'),
     Output('perf-anova-controls', 'style'),
     Output('perf-interaction-controls', 'style'),
     Output('perf-alpha-container', 'style')],
    [Input('perf-normality-check', 'value'),
     Input('perf-anova-check', 'value'),
     Input('perf-interaction-check', 'value'),
     Input('perf-hybrid-check', 'value')]
)
def toggle_performance_controls(norm_val, anova_val, inter_val, hybrid_val):
    norm_style = {'display': 'block'} if 'yes' in (norm_val or []) else {'display': 'none'}
    anova_style = {'display': 'block'} if 'yes' in (anova_val or []) else {'display': 'none'}
    inter_style = {'display': 'block'} if ('yes' in (inter_val or []) or 'yes' in (hybrid_val or [])) else {'display': 'none'}
    alpha_style = {'display': 'block'} if ('yes' in (norm_val or []) or 'yes' in (anova_val or [])) else {'display': 'none'}
    return norm_style, anova_style, inter_style, alpha_style

@app.callback(
    [Output('perf-normality-group', 'options'),
     Output('perf-normality-group-container', 'style'),
     Output('perf-anova-independents', 'options'),
     Output('perf-anova-independents', 'value'),
     Output('perf-interaction-x', 'options'),
     Output('perf-interaction-x', 'value'),
     Output('perf-interaction-line', 'options'),
     Output('perf-interaction-line', 'value'),
     Output('perf-interaction-facet', 'options'),
     Output('perf-interaction-facet', 'value'),
     Output('perf-interaction-check-container', 'style')],
    [Input('perf-protocol-dropdown', 'value')]
)
def update_perf_options(prot):
    norm_style = {'display': 'block'}
    inter_check_style = {'display': 'block'}
    if prot == 'C':
        inter_check_style = {'display': 'none'}
    if prot == 'A':
        norm_opts = [{'label': 'CV', 'value': 'CV'}, {'label': 'SV', 'value': 'SV'}, {'label': 'Ambos', 'value': 'Ambos'}]
        indep_opts = [{'label': 'Complexidade', 'value': 'Complexidade'}, {'label': 'Grupo', 'value': 'grupo'}, {'label': 'Overlap', 'value': 'Overlap'}]
        indep_val = ['Complexidade', 'grupo', 'Overlap']
    elif prot == 'B':
        norm_opts = [{'label': 'CF', 'value': 'CF'}, {'label': 'SF', 'value': 'SF'}, {'label': 'Ambos', 'value': 'Ambos'}]
        indep_opts = [{'label': 'Complexidade', 'value': 'Complexidade'}, {'label': 'Grupo', 'value': 'grupo'}]
        indep_val = ['Complexidade', 'grupo']
    else:
        norm_opts = [{'label': 'N/A', 'value': 'ALL'}]
        norm_style = {'display': 'none'}
        indep_opts = [{'label': 'Complexidade', 'value': 'Complexidade'}]
        indep_val = ['Complexidade']
        
    ix_opts = indep_opts
    ix_val = indep_opts[0]['value']
    il_val = indep_opts[1]['value'] if len(indep_opts) > 1 else indep_opts[0]['value']
    if_opts = [{'label': 'None', 'value': 'None'}] + indep_opts
    if_val = 'None'
    return norm_opts, norm_style, indep_opts, indep_val, ix_opts, ix_val, ix_opts, il_val, if_opts, if_val, inter_check_style

@app.callback(
    [Output('perf-normality-result-card', 'style'), Output('perf-normality-output', 'children'),
     Output('perf-anova-result-card', 'style'), Output('perf-anova-table-output', 'children'),
     Output('perf-interaction-result-card', 'style'), Output('perf-interaction-plot', 'figure'),
     Output('perf-hybrid-result-card', 'style'), Output('perf-hybrid-plot', 'figure'),
     Output('perf-controls-error', 'children')],
    [Input('perf-run-btn', 'n_clicks')],
    [State('perf-protocol-dropdown', 'value'), State('perf-normality-check', 'value'),
     State('perf-normality-var', 'value'), State('perf-normality-group', 'value'),
     State('perf-anova-check', 'value'), State('perf-anova-target', 'value'),
     State('perf-anova-independents', 'value'), State('perf-alpha-input', 'value'),
     State('perf-interaction-check', 'value'), State('perf-hybrid-check', 'value'),
     State('perf-interaction-x', 'value'), State('perf-interaction-y', 'value'),
     State('perf-interaction-line', 'value'), State('perf-interaction-facet', 'value'),
     State('perf-interaction-ylim', 'value')]
)
def run_performance_analysis(n_clicks, prot, norm_chk, norm_var, norm_group,
                            anova_chk, anova_var, anova_indeps, alpha,
                            inter_chk, hybrid_chk, inter_x, inter_y, inter_line, inter_facet, inter_ylim):
    if not n_clicks: raise PreventUpdate
    try:
        import os, ast
        from performance_engine import (run_normality_test, run_anova_typ3, plot_interactive_dot_sig,
                                        plot_interactive_interaction, plot_interactive_hybrid)
        csv_path = os.path.join(os.path.dirname(__file__), 'data', f'df_prot{prot}_performance.csv')
        if not os.path.exists(csv_path):
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f"Arquivo não encontrado: {csv_path}"
        df = pd.read_csv(csv_path)
        if prot == 'C' and 'Fase' in df.columns:
            df = df[df['Fase'] == 'Fase Execucao'].copy()
        alpha = float(str(alpha).replace(',', '.')) if alpha else 0.05
        norm_style, norm_out = {'display': 'none'}, ""
        anova_style, anova_tbl = {'display': 'none'}, ""
        inter_style, inter_fig = {'display': 'none'}, go.Figure()
        hybrid_style, hybrid_fig = {'display': 'none'}, go.Figure()
        
        if 'yes' in (norm_chk or []):
            df_norm = df.copy()
            if prot in ['A', 'B'] and norm_group != 'Ambos': df_norm = df_norm[df_norm['grupo'] == norm_group]
            norm_res, norm_err = run_normality_test(df_norm[norm_var], alpha=alpha)
            norm_style = {'display': 'block'}
            if norm_err: norm_out = html.Div(norm_err, className="alert alert-danger")
            else:
                s, k = norm_res['Shapiro-Wilk'], norm_res['Kolmogorov-Smirnov']
                def b(is_norm): return html.Span("✅ Normal", className="badge bg-success ms-2") if is_norm else html.Span("❌ Não Normal", className="badge bg-danger ms-2")
                norm_out = html.Div([
                    html.H5(f"Testando: {norm_var} | Grupo: {norm_group if prot in ['A','B'] else 'N/A'}", className="mb-3"),
                    html.Div([html.Strong("Shapiro-Wilk:"), html.Span(f" W = {s['stat']:.4f}, p = {s['p_value']:.4e}", className="ms-2"), b(s['is_normal'])], className="mb-2"),
                    html.Div([html.Strong("Kolmogorov-Smirnov:"), html.Span(f" D = {k['stat']:.4f}, p = {k['p_value']:.4e}", className="ms-2"), b(k['is_normal'])], className="mb-2")
                ])

        if 'yes' in (anova_chk or []):
            anova_res, anova_err = run_anova_typ3(df, anova_var, anova_indeps)
            anova_style = {'display': 'block'}
            if anova_err: anova_tbl = html.Div(anova_err, className="alert alert-danger")
            elif anova_res is not None:
                anova_res = anova_res.reset_index().rename(columns={'index': 'Termo', 'sum_sq': 'Soma dos Quadrados', 'PR(>F)': 'p-valor'})
                for col in ['Soma dos Quadrados', 'df', 'F', 'p-valor']:
                    if col in anova_res.columns: anova_res[col] = anova_res[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
                from dash import dash_table
                anova_tbl = html.Div([
                    html.P("👆 Clique em uma linha da tabela acima para atualizar o gráfico e os resultados do Post-Hoc abaixo.", className="text-muted small mb-2"),
                    dash_table.DataTable(
                        id='perf-anova-data-table', data=anova_res.to_dict('records'),
                        columns=[{'name': i, 'id': i} for i in anova_res.columns],
                        style_cell={'textAlign': 'left', 'padding': '10px'},
                        style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                        style_data_conditional=[{'if': {'filter_query': '{p-valor} < '+str(alpha), 'column_id': 'p-valor'}, 'backgroundColor': '#d4edda', 'fontWeight': 'bold'}]
                    )
                ])

        if 'yes' in (inter_chk or []):
            try: parsed_ylim = ast.literal_eval(inter_ylim) if inter_ylim else None
            except: parsed_ylim = None
            facet = inter_facet if inter_facet and inter_facet != 'None' else None
            fig, err = plot_interactive_interaction(df, inter_x, inter_line, inter_y, facet, parsed_ylim, alpha=alpha)
            inter_style, inter_fig = {'display': 'block'}, fig

        if 'yes' in (hybrid_chk or []):
            try: parsed_ylim = ast.literal_eval(inter_ylim) if inter_ylim else None
            except: parsed_ylim = None
            facet = inter_facet if inter_facet and inter_facet != 'None' else None
            fig_h, err_h = plot_interactive_hybrid(df, inter_x, inter_line, inter_y, facet, parsed_ylim, alpha=alpha)
            hybrid_style, hybrid_fig = {'display': 'block'}, fig_h

        return norm_style, norm_out, anova_style, anova_tbl, inter_style, inter_fig, hybrid_style, hybrid_fig, ""
    except Exception as e:
        import traceback; traceback.print_exc()
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f"Erro Crítico: {str(e)}"

@app.callback(
    [Output('perf-anova-dotplot', 'figure'),
     Output('perf-anova-heatmap', 'figure'),
     Output('perf-anova-heatmap-container', 'style'),
     Output('perf-anova-posthoc-output', 'children')],
    [Input('perf-anova-data-table', 'data'), Input('perf-anova-data-table', 'active_cell')],
    [State('perf-protocol-dropdown', 'value'), State('perf-anova-target', 'value'),
     State('perf-alpha-input', 'value'), State('perf-heatmap-check', 'value')]
)
def update_posthoc_plot(table_data, active_cell, prot, target_var, alpha, heatmap_chk):
    if not table_data: return go.Figure(), go.Figure(), {'display': 'none'}, "Execute a ANOVA primeiro."
    alpha = float(str(alpha).replace(',', '.')) if alpha else 0.05
    def get_p(r):
        try:
            p = r.get('p-valor')
            return float(p) if p and str(p).strip() else 1.0
        except: return 1.0
        
    idx = active_cell['row'] if active_cell else next((i for i, r in enumerate(table_data) if r.get('Termo') not in ['Intercept', 'Residual'] and get_p(r) < alpha), next((i for i, r in enumerate(table_data) if r.get('Termo') not in ['Intercept', 'Residual']), None))
    if idx is None: return go.Figure(), go.Figure(), {'display': 'none'}, "Nenhum fator analisável encontrado."
    source_str = table_data[idx]['Termo']
    import re; clean = source_str.replace("Q('", "").replace("')", ""); factors = re.findall(r'C\((.*?)\)', clean)
    if not factors:
        factors = [f.strip() for f in clean.split(':') if f.strip() not in ['Intercept', 'Residual']]
    if not factors: return go.Figure(), go.Figure(), {'display': 'none'}, f"Não é possível fazer Post-Hoc para a linha selecionada: {source_str}."
    import os, pandas as pd; from performance_engine import plot_interactive_dot_sig, plot_interactive_significance_heatmap
    csv_path = os.path.join(os.path.dirname(__file__), 'data', f'df_prot{prot}_performance.csv')
    df = pd.read_csv(csv_path)
    if prot == 'C' and 'Fase' in df.columns:
        df = df[df['Fase'] == 'Fase Execucao'].copy()
    gc = "_".join(factors); df[gc] = df[factors].astype(str).agg('_'.join, axis=1)
    
    fig_dot, sig, err = plot_interactive_dot_sig(df, gc, target_var, alpha=alpha, title=f"Post-Hoc: {gc} ({target_var}) [Fonte: {source_str}]", anova_table=table_data)
    
    fig_hm = go.Figure()
    hm_style = {'display': 'none'}
    if 'yes' in (heatmap_chk or []):
        fig_hm, err_hm = plot_interactive_significance_heatmap(df, target_var, gc, anova_table=table_data, alpha=alpha)
        if not err_hm:
            hm_style = {'display': 'block'}
            
    if err: return fig_dot, fig_hm, hm_style, html.Div(err, className="alert alert-warning")
    if not sig: return fig_dot, fig_hm, hm_style, html.Div("Nenhuma diferença significativa encontrada.", className="alert alert-info")
    from dash import dash_table; sdf = pd.DataFrame(sig)
    return fig_dot, fig_hm, hm_style, dash_table.DataTable(data=sdf.to_dict('records'), columns=[{'name': i, 'id': i} for i in sdf.columns], style_cell={'textAlign': 'left', 'padding': '5px'}, style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'})



if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
