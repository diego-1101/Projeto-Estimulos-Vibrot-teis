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
import os

# Load Quick Guide Content
try:
    with open('QUICK_GUIDE.md', 'r', encoding='utf-8') as f:
        quick_guide_content = f.read()
except FileNotFoundError:
    quick_guide_content = "Quick Guide not found. Please ensure QUICK_GUIDE.md is in the project root."

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
        color_s = meta['grupo'].astype(str)
        symbol_s = meta['grupo'].astype(str)
        
        # Safe getters 
        comp = meta.get('Complexidade', pd.Series('Unk', index=meta.index)).astype(str).str.replace(r'\.0$', '', regex=True)
        over = meta.get('Overlap', pd.Series('Unk', index=meta.index)).astype(str).str.replace(r'\.0$', '', regex=True)
        
        if mode == 'group':
            color_s = meta['grupo'].astype(str)
            symbol_s = color_s
        elif mode == 'complexity':
            color_s = 'C' + comp
            symbol_s = color_s
        elif mode == 'overlap':
            color_s = 'O' + over
            symbol_s = color_s
        elif mode == 'group_comp':
            color_s = meta['grupo'].astype(str)
            symbol_s = 'C' + comp
        elif mode == 'group_overlap':
            color_s = meta['grupo'].astype(str)
            symbol_s = 'O' + over
        elif mode == 'comp_overlap':
            color_s = 'C' + comp
            symbol_s = 'O' + over
        elif mode == 'all':
            color_s = meta['grupo'].astype(str) + '_C' + comp
            symbol_s = 'O' + over
        
        color_labels = color_s.tolist()
        symbol_labels = symbol_s.tolist()
    except KeyError as e:
        # Fallback if a column is miraculously missing
        print(f"KeyError: {e}. Falling back to default Group coloring.")
        color_labels = meta['grupo'].astype(str).tolist()
        symbol_labels = color_labels
        warning = True
        
    return color_labels, symbol_labels, warning

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

        html.Label("X (PSD Features)", className="control-label"),
        dcc.Dropdown(
            id={'type': 'x-mode-dropdown', 'index': panel_id},
            options=[
                {'label': 'PSD trecho completo', 'value': 'psd_full'},
                {'label': 'PSD estratificada por bandas', 'value': 'psd_bands'},
                {'label': 'PSD estratificada (normalizada)', 'value': 'psd_bands_norm'}
            ],
            value='psd_bands_norm',
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
                    {'label': 'Complexity', 'value': 'complexity'},
                    {'label': 'Overlap (Prot A)', 'value': 'overlap'}
                ],
                value='complexity',
                className="mb-3 dash-dropdown"
            ),
        ]),
        
        html.Hr(),
    ], className="comparison-panel mb-3" if panel_id == 2 else "mb-3")

def run_single_analysis(protocol, groups_selected, method, x_mode, y_cols, domain, axes, n_dims, theme='light', color_by='group', supervision_by='complexity'):
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
        X = build_X(df, x_mode)
        Y = build_Y(df, y_cols)
        
        # Build supervision labels
        # 1. Labels for Visual Coloring
        color_labels, symbol_labels, _ = build_supervision_labels(meta, protocol, color_by)
        
        # 2. Labels for Mathematical Supervision (CDA/LDA)
        # Simplified: user selects Complexity or Overlap as the column name
        if supervision_by == 'overlap' and 'Overlap' in meta.columns:
            math_labels = meta['Overlap'].astype(str).tolist()
            had_warning = False
        else:
            # Default Complexity
            math_labels = meta.get('Complexidade', pd.Series('Unk', index=meta.index)).astype(str).str.replace(r'\.0$', '', regex=True).tolist()
            had_warning = False

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
        
        # --- Projected ANOVA ---
        # Calculate ANOVA on the first plotted dimension vs color labels
        # we must drop NaNs for the stat test
        stat_df = plot_df[[axis_names[0], 'color_label']].dropna()
        try:
            if not stat_df.empty and len(stat_df['color_label'].unique()) > 1:
                from anova_engine import compute_anova_and_plot
                anova_fig, anova_stats = compute_anova_and_plot(stat_df, axis_names[0], 'color_label')
                anova_fig.update_layout(title=f"ANOVA: {axis_names[0]}")
                if theme == 'dark':
                    anova_fig.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                anova_res_text = f"F: {anova_stats.get('F', 0):.2f} | p-value: {anova_stats.get('p_value', 1):.4f}"
            else:
                anova_fig, anova_res_text = go.Figure(), "ANOVA: Insufficient groups/data"
        except Exception as ae:
            anova_fig, anova_res_text = go.Figure(), f"ANOVA Error: {str(ae)}"

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
            
        params = {
             'color': 'color_label',
             'symbol': 'symbol_label',
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

        return fig, html.Div(stats_content or "No stats evaluated"), anova_fig, anova_res_text
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Error: {str(e)}")
        return error_fig, html.Div(f"Error: {str(e)}"), []

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
                    html.Button("📖 Quick Guide", id='open-guide-btn', className='btn btn-sm btn-outline-info')
                ], style={'position': 'absolute', 'top': '10px', 'left': '15px', 'zIndex': 500}),
                dcc.Graph(id='plot-1', style={'height': '600px'})
            ], className="card mb-3", style={'position': 'relative'}),
            html.Div([html.H4("Statistics"), html.Div(id='stats-1')], className="card mb-3"),
            html.Div([
                html.H4("ANOVA Test"),
                html.Div(id='anova-stats-1', className="mb-2"),
                dcc.Graph(id='anova-plot-1', style={'height': '500px'})
            ], className="card")
        ]),
        
        html.Div(id='comparison-view', style={'display': 'none'}, children=[
            html.Div(className="comparison-container", children=[
                html.Div([
                    html.Div([
                        html.Div([
                            html.Button("📖 Quick Guide", id='open-guide-btn-left', className='btn btn-sm btn-outline-info')
                        ], style={'position': 'absolute', 'top': '10px', 'left': '15px', 'zIndex': 500}),
                        dcc.Graph(id='plot-left', style={'height': '500px'})
                    ], className="card mb-3", style={'position': 'relative'}),
                    html.Div([html.H5("Stats 1"), html.Div(id='stats-left')], className="card mt-3 mb-3"),
                    html.Div([
                        html.H5("ANOVA 1"),
                        html.Div(id='anova-stats-left', className="mb-2"),
                        dcc.Graph(id='anova-plot-left', style={'height': '400px'})
                    ], className="card")
                ]),
                html.Div([
                    html.Div([
                        html.Div([
                            html.Button("📖 Quick Guide", id='open-guide-btn-right', className='btn btn-sm btn-outline-info')
                        ], style={'position': 'absolute', 'top': '10px', 'left': '15px', 'zIndex': 500}),
                        dcc.Graph(id='plot-right', style={'height': '500px'})
                    ], className="card mb-3", style={'position': 'relative'}),
                    html.Div([html.H5("Stats 2"), html.Div(id='stats-right')], className="card mt-3 mb-3"),
                    html.Div([
                        html.H5("ANOVA 2"),
                        html.Div(id='anova-stats-right', className="mb-2"),
                        dcc.Graph(id='anova-plot-right', style={'height': '400px'})
                    ], className="card")
                ])
            ])
        ])
    ], className="main-content", id='main-content')
])

# --- Callbacks ---

@app.callback(
    Output('quick-guide-modal', 'style'),
    [Input('open-guide-btn', 'n_clicks'),
     Input('open-guide-btn-left', 'n_clicks'),
     Input('open-guide-btn-right', 'n_clicks'),
     Input('close-guide-btn', 'n_clicks')],
    State('quick-guide-modal', 'style'),
    prevent_initial_call=True
)
def toggle_quick_guide(btn1, btn2, btn3, close_clicks, current_style):
    from dash import ctx
    if not ctx.triggered:
        return {'display': 'none'}
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id in ['open-guide-btn', 'open-guide-btn-left', 'open-guide-btn-right']:
        return {'display': 'block'}
    elif trigger_id == 'close-guide-btn':
        return {'display': 'none'}
    return current_style

@app.callback(
    Output('math-model-container', 'children'),
    [Input('protocol-dropdown', 'value'),
     Input({'type': 'x-mode-dropdown', 'index': 1}, 'value'),
     Input({'type': 'y-checklist', 'index': 1}, 'value')]
)
def update_math_model(prot, x_mode, y_cols):
    try:
        # Load caching internally so it doesn't freeze the UI
        if prot not in data_cache:
            df, meta = load_data(protocol=prot)
            data_cache[prot] = (df, meta)
        else:
            df, meta = data_cache[prot]
            
        if not y_cols:
            y_cols = ['Desempenho']
            
        X = build_X(df, x_mode)
        Y = build_Y(df, y_cols)
        
        T = X.shape[0] if not X.empty else "T"
        F = X.shape[1] if not X.empty else "F"
        C = Y.shape[1] if not Y.empty else 1
        
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
* **F = {F}**: Número de *features* em $X$ (modo: {x_mode}).
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
     Output('info-panel', 'children')],
    Input('run-btn', 'n_clicks'),
    [State('protocol-dropdown', 'value'),
     State('group-checklist', 'value'),
     State({'type': 'method-dropdown', 'index': 1}, 'value'),
     State({'type': 'x-mode-dropdown', 'index': 1}, 'value'),
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
def update_single_analysis(n, prot, groups, meth, x_mode, y_cols, dom, dims, color, supervision_by, theme, ax1, ax2, ax3, comp):
    if n == 0 or 'yes' in comp:
        fig = go.Figure()
        fig.update_layout(title="Click Run Analysis")
        return fig, "No data", go.Figure(), "", html.P("Ready")
    
    axes = [ax1, ax2, ax3]
    fig, stats, anova_fig, anova_res = run_single_analysis(prot, groups, meth, x_mode, y_cols, dom, axes, dims, theme, color, supervision_by)
    
    info = html.Div([
        html.P([html.Strong("Protocol: "), prot]),
        html.P([html.Strong("Groups: "), ", ".join(groups) if groups else "All"]),
        html.H6(f"Method: {meth}", style={'marginTop': '10px'}),
    ])
    return fig, stats, anova_fig, anova_res, info

@app.callback(
    [Output('plot-left', 'figure'), Output('stats-left', 'children'),
     Output('anova-plot-left', 'figure'), Output('anova-stats-left', 'children'),
     Output('plot-right', 'figure'), Output('stats-right', 'children'),
     Output('anova-plot-right', 'figure'), Output('anova-stats-right', 'children')],
    Input('run-btn', 'n_clicks'),
    [State('protocol-dropdown', 'value'),
     State('group-checklist', 'value'),
     State({'type': 'method-dropdown', 'index': ALL}, 'value'),
     State({'type': 'x-mode-dropdown', 'index': ALL}, 'value'),
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
def update_comparison(n, prot, groups, methods, x_modes, y_cols_lists, doms, dims, colors, supervisions, theme, comp, ax1s, ax2s, ax3s):
    fig = go.Figure()
    fig.update_layout(title="Enable comparison mode")
    
    if n == 0 or 'yes' not in comp or len(methods) < 2:
        return fig, "Waiting...", go.Figure(), "", fig, "Waiting...", go.Figure(), ""
        
    # Analysis 1
    axes1 = [ax1s[0], ax2s[0], ax3s[0]]
    fig1, stats1, anova_fig1, anova_res1 = run_single_analysis(
        prot, groups, methods[0], x_modes[0], y_cols_lists[0], doms[0], axes1, dims, theme, colors[0], supervisions[0]
    )
    
    # Analysis 2
    axes2 = [ax1s[1], ax2s[1], ax3s[1]]
    fig2, stats2, anova_fig2, anova_res2 = run_single_analysis(
        prot, groups, methods[1], x_modes[1], y_cols_lists[1], doms[1], axes2, dims, theme, colors[1], supervisions[1]
    )
    
    return fig1, stats1, anova_fig1, anova_res1, fig2, stats2, anova_fig2, anova_res2

# Expose server for Vercel
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
