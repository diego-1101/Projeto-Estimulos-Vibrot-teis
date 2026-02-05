import dash
from dash import dcc, html, Input, Output, State, ALL
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Import our modules
from data_loader import load_and_preprocess_data
from analysis_engine import compute_embedding

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=['https://bootswatch.com/5/flatly/bootstrap.min.css'])
app.title = "EEG PSD Dashboard"

# Data cache
data_cache = {}

# Theme state (global for simplicity)
current_theme = 'light'

# --- Helper Functions ---

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
        
        html.Label("Covariance Mode", className="control-label"),
        dcc.RadioItems(
            id={'type': 'covariance-mode', 'index': panel_id},
            options=[
                {'label': ' Auto-covariance', 'value': 'auto'},
                {'label': ' Cross-covariance', 'value': 'cross'}
            ],
            value='auto',
            className="mb-3"
        ),
        
        html.Label("Data Domain", className="control-label"),
        dcc.Dropdown(
            id={'type': 'domain-dropdown', 'index': panel_id},
            options=[
                {'label': 'PSD Features', 'value': 'psd'},
                {'label': 'Behavioral', 'value': 'bx'}
            ],
            value='psd',
            className="mb-3 dash-dropdown"
        ),
        
        html.Label("Dimensions", className="control-label"),
        dcc.RadioItems(
            id={'type': 'dimensions-radio', 'index': panel_id},
            options=[
                {'label': ' 2D', 'value': 2},
                {'label': ' 3D', 'value': 3}
            ],
            value=2,
            className="mb-3"
        ),

        html.Label("Color By", className="control-label"),
        dcc.Dropdown(
            id={'type': 'color-dropdown', 'index': panel_id},
            options=[
                {'label': 'Group (CV/SV)', 'value': 'group'},
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
    ], className="comparison-panel mb-3" if panel_id == 2 else "mb-3")

def run_single_analysis(protocol, method, cov_mode, domain, n_dims, theme='light', color_by='group'):
    """Run analysis and return figure, stats."""
    try:
        if protocol not in data_cache:
            X_psd, X_bx, meta, feature_names = load_and_preprocess_data(protocol=protocol)
            data_cache[protocol] = (X_psd, X_bx, meta, feature_names)
        
        X_psd, X_bx, meta, feature_names = data_cache[protocol]
        X = X_psd if domain == 'psd' else X_bx
        Y_labels = meta['grupo']
        Y_continuous = X_bx if cov_mode == 'cross' else None
        
        embedding, stats = compute_embedding(
            X=X, Y_labels=Y_labels, Y_continuous=Y_continuous,
            method=method, covariance_mode=cov_mode, n_components=n_dims
        )
        
        plot_df = pd.concat([embedding, meta.reset_index(drop=True)], axis=1)
        
        # --- Coloring Logic ---
        # Ensure columns exist before using them (handle missing data gracefully)
        has_complex = 'Complexidade' in plot_df.columns
        has_overlap = 'Overlap' in plot_df.columns
        
        if color_by == 'complexity' and has_complex:
            plot_df['color_label'] = plot_df['Complexidade'].astype(str)
            title_suffix = "Complexity"
        elif color_by == 'overlap' and has_overlap:
            plot_df['color_label'] = plot_df['Overlap'].astype(str)
            title_suffix = "Overlap"
        elif color_by == 'group_comp' and has_complex:
            plot_df['color_label'] = plot_df['grupo'] + '_C' + plot_df['Complexidade'].astype(str)
            title_suffix = "Group + Complexity"
        elif color_by == 'group_overlap' and has_overlap:
            plot_df['color_label'] = plot_df['grupo'] + '_O' + plot_df['Overlap'].astype(str)
            title_suffix = "Group + Overlap"
        elif color_by == 'comp_overlap' and has_complex and has_overlap:
            plot_df['color_label'] = 'C' + plot_df['Complexidade'].astype(str) + '_O' + plot_df['Overlap'].astype(str)
            title_suffix = "Complexity + Overlap"
        elif color_by == 'all' and has_complex and has_overlap:
            plot_df['color_label'] = plot_df['grupo'] + '_C' + plot_df['Complexidade'].astype(str) + '_O' + plot_df['Overlap'].astype(str)
            title_suffix = "All Factors"
        else:
            # Default to group or fallback
            plot_df['color_label'] = plot_df['grupo']
            title_suffix = "Group"
            if color_by != 'group':
                title_suffix += " (Data Warning)"

        # Define hover columns properly
        hover_cols = ['ID', 'grupo']
        if has_complex: hover_cols.append('Complexidade')
        if has_overlap: hover_cols.append('Overlap')

        # Colors: We drop specific map to allow Plotly to assign distinct colors for many categories
        # But we keep specific map for simple Group case
        color_map = None
        if color_by == 'group':
            color_map = {'CV': '#2ecc71', 'SV': '#e74c3c', 'CF': '#2ecc71', 'SF': '#e74c3c'}

        if n_dims == 2:
            fig = px.scatter(
                plot_df, x='C1', y='C2', color='color_label',
                hover_data=hover_cols,
                color_discrete_map=color_map,
                title=f"{method} - {cov_mode} - {domain} - {title_suffix}"
            )
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
        else:
            fig = px.scatter_3d(
                plot_df, x='C1', y='C2', z='C3', color='color_label',
                hover_data=hover_cols,
                color_discrete_map=color_map,
                title=f"{method} - {cov_mode} - {domain} - {title_suffix}"
            )
            fig.update_traces(marker=dict(size=6))
        
        if theme == 'dark':
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#2d2d2d',
                plot_bgcolor='#2d2d2d',
                font=dict(color='#e0e0e0')
            )
        else:
            fig.update_layout(template='plotly_white')
        
        stats_content = []
        if 'explained_variance' in stats:
            var_text = ", ".join([f"C{i+1}: {v*100:.1f}%" for i, v in enumerate(stats['explained_variance'])])
            stats_content.append(html.P([html.Strong("Variance: "), var_text]))
        if 'canonical_correlations' in stats:
            corr_text = ", ".join([f"r{i+1}: {c:.3f}" for i, c in enumerate(stats['canonical_correlations'])])
            stats_content.append(html.P([html.Strong("Correlations: "), corr_text]))
        
        return fig, html.Div(stats_content or "No stats")
        
    except Exception as e:
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Error: {str(e)}")
        return error_fig, html.Div(f"Error: {str(e)}")

# --- Layout ---
app.layout = html.Div([
    html.Button("🌙", id='theme-toggle', className='theme-toggle', n_clicks=0),
    dcc.Store(id='theme-store', data='light'),
    html.Div(id='theme-injector', style={'display': 'none'}),
    
    html.Div([
        html.H2("EEG PSD Dashboard", className="text-primary mb-4"),
        html.H5("Multivariate Analysis", className="text-muted mb-4"),
        html.Hr(),
        
        html.Label("Protocol", className="control-label"),
        dcc.Dropdown(
            id='protocol-dropdown',
            options=[
                {'label': 'Protocol A', 'value': 'A'},
                {'label': 'Protocol B', 'value': 'B'}
            ],
            value='A',
            className="mb-3 dash-dropdown"
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
        
        html.Button("Run Analysis", id='run-btn', className="btn btn-primary w-100 mt-3"),
        html.Hr(),
        html.Div(id='info-panel', className="card p-3 mt-3")
        
    ], className="sidebar"),
    
    html.Div([
        html.Div(id='single-view', children=[
            html.Div([dcc.Graph(id='plot-1', style={'height': '600px'})], className="card mb-3"),
            html.Div([html.H4("Statistics"), html.Div(id='stats-1')], className="card")
        ]),
        
        html.Div(id='comparison-view', style={'display': 'none'}, children=[
            html.Div(className="comparison-container", children=[
                html.Div([
                    html.Div([dcc.Graph(id='plot-left', style={'height': '500px'})], className="card"),
                    html.Div([html.H5("Stats 1"), html.Div(id='stats-left')], className="card mt-3")
                ]),
                html.Div([
                    html.Div([dcc.Graph(id='plot-right', style={'height': '500px'})], className="card"),
                    html.Div([html.H5("Stats 2"), html.Div(id='stats-right')], className="card mt-3")
                ])
            ])
        ])
    ], className="main-content", id='main-content')
])

# --- Callbacks ---

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

# Clientside callback to apply theme to body
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

@app.callback(
    Output({'type': 'domain-dropdown', 'index': ALL}, 'disabled'),
    [Input({'type': 'covariance-mode', 'index': ALL}, 'value'),
     Input({'type': 'method-dropdown', 'index': ALL}, 'value')]
)
def disable_domains(covs, methods):
    return [c == 'cross' and m in ['PLS', 'CDA'] for c, m in zip(covs, methods)]

@app.callback(
    [Output('plot-1', 'figure'), Output('stats-1', 'children'), Output('info-panel', 'children')],
    Input('run-btn', 'n_clicks'),
    [State('protocol-dropdown', 'value'),
     State({'type': 'method-dropdown', 'index': 1}, 'value'),
     State({'type': 'covariance-mode', 'index': 1}, 'value'),
     State({'type': 'domain-dropdown', 'index': 1}, 'value'),
     State({'type': 'dimensions-radio', 'index': 1}, 'value'),
     State({'type': 'color-dropdown', 'index': 1}, 'value'),
     State('theme-store', 'data'),
     State('comparison-toggle', 'value')],
    prevent_initial_call=True
)
def update_single(n, prot, meth, cov, dom, dims, color, theme, comp):
    if n == 0 or 'yes' in comp:
        fig = go.Figure()
        fig.update_layout(title="Click Run Analysis")
        return fig, "No data", html.P("Ready")
    
    fig, stats = run_single_analysis(prot, meth, cov, dom, dims, theme, color)
    info = html.Div([
        html.P([html.Strong("Protocol: "), prot]),
        html.P([html.Strong("Method: "), meth]),
        html.P([html.Strong("Color By: "), color])
    ])
    return fig, stats, info

@app.callback(
    [Output('plot-left', 'figure'), Output('stats-left', 'children'),
     Output('plot-right', 'figure'), Output('stats-right', 'children')],
    Input('run-btn', 'n_clicks'),
    [State('protocol-dropdown', 'value'),
     State({'type': 'method-dropdown', 'index': ALL}, 'value'),
     State({'type': 'covariance-mode', 'index': ALL}, 'value'),
     State({'type': 'domain-dropdown', 'index': ALL}, 'value'),
     State({'type': 'dimensions-radio', 'index': ALL}, 'value'),
     State({'type': 'color-dropdown', 'index': ALL}, 'value'),
     State('theme-store', 'data'),
     State('comparison-toggle', 'value')],
    prevent_initial_call=True
)
def update_comparison(n, prot, methods, covs, doms, dims, colors, theme, comp):
    fig = go.Figure()
    fig.update_layout(title="Enable comparison mode")
    
    if n == 0 or 'yes' not in comp or len(methods) < 2:
        return fig, "No data", fig, "No data"
    
    # Analysis 1
    m1, c1, d1, dim1, col1 = methods[0], covs[0], doms[0], dims[0], colors[0]
    # Analysis 2
    m2, c2, d2, dim2, col2 = methods[1], covs[1], doms[1], dims[1], colors[1]
    
    fig1, stats1 = run_single_analysis(prot, m1, c1, d1, dim1, theme, col1)
    fig2, stats2 = run_single_analysis(prot, m2, c2, d2, dim2, theme, col2)
    return fig1, stats1, fig2, stats2

# Expose server for Vercel
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
